import math
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import utils
from bert import BertForMaskedLM
from text.symbols import symbols
from data_utils import TextLoader, DataCollatorForLanguageModeling

device = "cuda" if torch.cuda.is_available() else "cpu"
global_step = 0


def main():
    hps = utils.get_hparams()
    run(hps)


def run(hps):
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    train_dataset = TextLoader(hps.data.training_files, hps.data)
    collate_fn = DataCollatorForLanguageModeling()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size,
    )

    eval_dataset = TextLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=hps.train.batch_size * 2,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = BertForMaskedLM(
        n_vocab=len(symbols),
        out_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    epoch_str = 1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            epoch,
            hps,
            model,
            optim,
            scheduler,
            scaler,
            [train_loader, eval_loader],
            logger,
            [writer, writer_eval],
        )
        scheduler.step()


def train_and_evaluate(
    epoch, hps, model, optim, scheduler, scaler, loaders, logger, writers
):
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    model.train()
    for batch_idx, (text_padded, text_lengths, labels) in enumerate(train_loader):
        text_padded = text_padded.to(device)
        text_lengths = text_lengths.to(device)
        labels = labels.to(device)

        with autocast(enabled=hps.train.fp16_run):
            loss, logits, hidden_states = model(text_padded, text_lengths, labels)

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        scaler.step(optim)
        scaler.update()

        if global_step % hps.train.log_interval == 0:
            lr = optim.param_groups[0]["lr"]
            losses = [loss]
            logger.info(
                "Train Epoch: {} [{:.0f}%]".format(
                    epoch, 100.0 * batch_idx / len(train_loader)
                )
            )
            logger.info([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {
                "loss": loss,
                "learning_rate": lr,
            }
            utils.summarize(
                writer=writer,
                global_step=global_step,
                scalars=scalar_dict,
            )

        if global_step % hps.train.eval_interval == 0 and global_step != 0:
            evaluate(hps, model, eval_loader, logger, writer_eval)
            utils.save_checkpoint(
                model,
                optim,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "BERT_{}.pth".format(global_step)),
            )
        global_step += 1

    logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, model, eval_loader, logger, writer_eval):
    global global_step

    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (text_padded, text_lengths, labels) in enumerate(eval_loader):
            text_padded = text_padded.to(device)
            text_lengths = text_lengths.to(device)
            labels = labels.to(device)

            loss, logits, hidden_states = model(text_padded, text_lengths, labels)
            losses.append(loss.repeat(hps.train.batch_size))

    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    perplexity = math.exp(eval_loss)

    logger.info("Validation Loss: {:.3f}, Perplexity: {:.3f}".format(loss, perplexity))

    scalar_dict = {
        "eval_loss": loss,
        "eval_perplexity": perplexity,
    }
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        scalars=scalar_dict,
    )
    model.train()


if __name__ == "__main__":
    main()
