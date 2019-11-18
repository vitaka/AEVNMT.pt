import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import aevnmt.aevnmt_helper as aevnmt_helper
from aevnmt.components import ancestral_sample
from aevnmt.data import batch_to_sentences, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ParallelDatasetFlippedView

from itertools import cycle, chain

from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from pathlib import Path
from tensorboardX import SummaryWriter

from aevnmt.train_utils import load_data, load_vocabularies
from aevnmt.train_utils import StepCounter, CheckPoint
from aevnmt.opt_utils import construct_optimizers, lr_scheduler_step, take_optimizer_step, RequiresGradSwitch
from aevnmt.hparams import Hyperparameters
from aevnmt.data import BucketingParallelDataLoader, BucketingTextDataLoader
from aevnmt.data import PAD_TOKEN
from aevnmt.data.utils import create_noisy_batch
from aevnmt.models import initialize_model

from collections import defaultdict, deque
import numpy as np


def count_number_of_batches(iterator):
    n = 0
    for _ in iterator:
        n += 1
    return n

class RunningStats:

    def __init__(self, update_factor=0.05):
        self.update_factor = update_factor
        self.avg = 0.
        self.std = 1.

    def update(self, avg, std):
        self.avg = self.update_factor * avg + (1 - self.update_factor) * self.avg
        self.std = max(1., self.update_factor * std + (1 - self.update_factor) * self.std)

class Tracker:

    def __init__(self, max_length=100):
        self.stats = defaultdict(lambda: deque([]))
        self.max_length = max_length

    def update(self, key, value):
        d = self.stats[key]
        d.append(value)
        if len(d) > self.max_length:
            d.popleft()

    def mean(self, key):
        return np.mean(self.stats[key])

    def sum(self, key):
        return np.sum(self.stats[key])

    def avg(self, key, steps):
        return self.sum(key) / (self.sum(steps) + 1e-6)


def create_model(hparams, vocab_src, vocab_tgt):
    if hparams.model_type == "cond_nmt":
        model = nmt_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = nmt_helper.train_step
        validate_fn = nmt_helper.validate
        translate_fn = nmt_helper.translate
        sample_fn=None
    elif hparams.model_type == "aevnmt":
        model = aevnmt_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = aevnmt_helper.train_step
        validate_fn = aevnmt_helper.validate
        translate_fn = aevnmt_helper.translate
        sample_fn = aevnmt_helper.re_sample
    else:
        raise Exception(f"Unknown model_type: {hparams.model_type}")

    return model, train_fn, validate_fn, translate_fn, sample_fn



def mono_vae_loss(
        model, hparams, inputs, noisy_inputs, targets, seq_len, qz, z, KL_weight, step,
        state=dict(), writer=None, title="SenVAE"):

    # [max_length, batch_size, vocab_size]
    lm_logits = model.run_language_model(noisy_inputs, z)
    # [batch_size, max_length, vocab_size]
    lm_logits = lm_logits.permute(0, 2, 1)
    log_prob = -F.cross_entropy(
        lm_logits, targets, ignore_index=model.pad_idx, reduction="none").sum(dim=1)

    pz = model.prior()
    KL = torch.distributions.kl.kl_divergence(qz, pz)

    # [batch_size]
    KL = torch.distributions.kl.kl_divergence(qz, pz)
    raw_KL = KL * 1

    #TODO: x side losses

    #TODO: check consistency with AEVNMT
    if hparams.KL_free_nats > 0:
        KL = torch.clamp(KL, min=hparams.KL_free_nats)
    KL *= KL_weight

    terms = dict()
    terms['raw_KL'] = raw_KL
    terms['KL'] = KL
    terms['LL'] = log_prob
    terms['loss'] = -(log_prob - KL)

    if writer:
        writer.add_scalar('%s/KL' % title, raw_KL.mean(), step)
        writer.add_scalar('%s/LL' % title, log_prob.mean(), step)
        writer.add_scalar('%s/ELBO' % title, (log_prob - raw_KL).mean(), step)

    return terms


def aevnmt_bilingual_step_xy(model, hparams,x_in, x_out, seq_mask_x, seq_len_x, y_in, y_out, seq_mask_y, seq_len_y,step, optimizers,lr_schedulers,KL_weight,tracker,writer=None,title="bilingual/xy", synthetic_x=False):
    #TODO: create shuf inputs later
    x_shuf_in=x_shuf_out=seq_mask_x_shuf=seq_len_x_shuf=noisy_x_shuf_in=None

    loss_terms = aevnmt_helper.train_step(
            model, x_in, x_out, seq_mask_x, seq_len_x, x_in,
            y_in, y_out, seq_mask_y, seq_len_y, y_in,
            x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in,
            hparams,
            step, summary_writer=writer, synthetic_x=synthetic_x)
    loss=loss_terms["loss"]

    # Backpropagate and update gradients.
    loss.backward()
    if hparams.max_gradient_norm > 0:
        # TODO: do we need separate norms?
        nn.utils.clip_grad_norm_(model.parameters(),
                                 hparams.max_gradient_norm)
    optimizers["gen"].step()
    if "inf_z" in optimizers: optimizers["inf_z"].step()
    # Zero the gradient buffer.
    optimizers["gen"].zero_grad()
    if "inf_z" in optimizers: optimizers["inf_z"].zero_grad()

    # Update the learning rate scheduler if needed.
    lr_scheduler_step(lr_schedulers, hparams)

     # These are for logging, thus we use raw KL
    ELBO = loss_terms['lm_log_likelihood'] + loss_terms['tm_log_likelihood'] - loss_terms['raw_KL']
    ELBO = ELBO.mean()
    KL = loss_terms['raw_KL'].mean()

    if writer:
        writer.add_scalar(f'{title}/loss', loss, step)
        # xy terms
        writer.add_scalar(f'{title}/ELBO', ELBO, step)
        writer.add_scalar(f'{title}/LM', loss_terms['lm_log_likelihood'].mean(), step)
        writer.add_scalar(f'{title}/TM', loss_terms['tm_log_likelihood'].mean(), step)
        writer.add_scalar(f'{title}/KL', KL, step)

    # Update statistics.
    tracker.update('ELBO', ELBO.item() * x_in.size(0))
    tracker.update('num_tokens', (seq_len_x.sum() + seq_len_y.sum()).item())
    tracker.update('num_sentences', x_in.size(0))

    step += 1

def senvae_monolingual_step_x(
        model,
        inputs, noisy_inputs, targets, seq_mask, seq_len,
        step, optimizers, KL_weight,
        hparams, tracker, writer=None, title='SenVAE'):

    # Infer q(z|x)
    qz = model.approximate_posterior(inputs, seq_mask, seq_len,None,None,None)
    # [B, dz]
    z = qz.rsample()
    # [B]
    if writer:
        writer.add_histogram("posterior-x/z", z, step)
        for param_name, param_value in get_named_params(qz):
            writer.add_histogram("posterior-x/%s" % param_name, param_value, step)

    # We will need to sample from q(x|z,y) where q(x|z,y) := p(x|z,y, theta_2)
    #  thus we start by encoding (z,y) with theta_2
    encoder_outputs, encoder_final = model.encode(noisy_inputs, seq_len, z)
    hidden = model.init_decoder(encoder_outputs, encoder_final, z)

    # We may update p(y|theta_2) on a SenVAE objective

    mono_vae_terms = mono_vae_loss(
        model=model, hparams=hparams,
        inputs=inputs, noisy_inputs=noisy_inputs, targets=targets, seq_len=seq_len,
        qz=qz, z=z,
        KL_weight=KL_weight, step=step,
        writer=writer, title=title
    )
    negative_elbo = mono_vae_terms['loss']
    negative_elbo.mean().backward(retain_graph=True)

    if hparams.max_gradient_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.lm_parameters(),
                                 max_norm=hparams.max_gradient_norm,
                                 norm_type=float("inf"))

    optimizers['gen'].step()
    optimizers['inf_z'].step()
    optimizers['gen'].zero_grad()
    optimizers['inf_z'].zero_grad()

    # Update statistics.
    ELBO = (mono_vae_terms['LL'] - mono_vae_terms['raw_KL'])
    tracker.update('SenVAE/ELBO', ELBO.sum().item())
    tracker.update('num_tokens', seq_len.sum().item())
    tracker.update('num_sentences', inputs.size(0))


def aevnmt_monolingual_step(model, vocab_src,
                            y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
                            hparams, step, device,
                            optimizers, KL_weight,
                            tracker: Tracker,
                            reward_stats=None,
                            writer=None,
                            gather_examples=None,
                            title='monolingual'):
    # Infer q(z|y)
    #TODO: name the 3 approximate posteriors
    # q(z|y)
    qz = model.approximate_posterior(None,None,None,y_in, seq_mask_y, seq_len_y)
    # [B, dz]
    z = qz.rsample()
    # [B]
    log_qz = qz.log_prob(z).sum(dim=1)
    if writer:
        writer.add_histogram("posterior-y/z", z, step)
        for param_name, param_value in get_named_params(qz):
            writer.add_histogram("posterior-y/%s" % param_name, param_value, step)

    hidden = model.init_lm(z)

    sample_dict = sampling_decode(model.language_model, model.language_model.embed,
                                   model.lm_generate, hidden,
                                   None, None,
                                   hparams.batch_size,None, vocab_src[SOS_TOKEN], vocab_src[EOS_TOKEN],
                                   vocab_src[PAD_TOKEN], hparams.max_decoding_length,hparams.sample_decoding_nucleus_p, z if hparams.feed_z else None)

    # Here we convert the sample to a batch of inputs
    raw_hypothesis = sample_dict['sample']

    #TODO: convert to strings only if there is a writer?
    hypothesis = batch_to_sentences(raw_hypothesis, vocab_src)
    if gather_examples is not None:
        gather_examples['sampled_x'] = hypothesis
    x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in = create_noisy_batch(
        hypothesis, vocab_src, device, word_dropout=0.)

    aevnmt_bilingual_step_xy(model, hparams,x_in, x_out, seq_mask_x, seq_len_x, y_in, y_out, seq_mask_y, seq_len_y,step, optimizers,lr_schedulers,KL_weight,tracker,writer=None,title="monolingual/y", synthetic_x=True):



def bilingual_loss(
        model, q_z,
        x_in, x_out, seq_mask_x, seq_len_x,
        y_in, y_out, seq_mask_y, seq_len_y,
        hparams, step,
        KL_weight, tracker,
        writer=None, title="bilingual":
    """Returns the loss for a given model on bilingual data"""

    # Compute the translation and language model logits.
    z = q_z.rsample()
    tm_logits, lm_logits, _ = model(x_in, seq_mask_x, seq_len_x, y_in, z)

    if writer:
        writer.add_histogram(f"{title}/z", z, step)

    # Compute the loss.
    loss_terms = model.loss(
        tm_logits, lm_logits, y_out, x_out, q_z,
        free_nats=hparams.KL_free_nats,
        KL_weight=KL_weight,
        reduction="mean")
    loss = loss_terms['loss']

    # These are for logging, thus we use raw KL
    ELBO = loss_terms['lm_log_likelihood'] + loss_terms['tm_log_likelihood'] - loss_terms['raw_KL']
    ELBO = ELBO.mean()
    KL = loss_terms['raw_KL'].mean()

    if writer:
        writer.add_scalar(f'{title}/loss', loss, step)
        # xy terms
        writer.add_scalar(f'{title}/ELBO', ELBO, step)
        writer.add_scalar(f'{title}/LM', loss_terms['lm_log_likelihood'].mean(), step)
        writer.add_scalar(f'{title}/TM', loss_terms['tm_log_likelihood'].mean(), step)
        writer.add_scalar(f'{title}/KL', KL, step)

    # Update statistics.
    tracker.update('ELBO', ELBO.item() * x_in.size(0))
    tracker.update('num_tokens', (seq_len_x.sum() + seq_len_y.sum()).item())
    tracker.update('num_sentences', x_in.size(0))

    return loss


def bilingual_step(
        model, q_z, hparams,
        x_in, x_out, seq_mask_x, seq_len_x,
        y_in, y_out, seq_mask_y, seq_len_y,
        step, optimizers, KL_weight, regularizer,
        tracker, writer=None, title="bilingual"):
    """Take a step towards minimising the loss"""

    loss = bilingual_loss(
        model=model, q_z=q_z,
        x_in=x_in, x_out=x_out, seq_mask_x=seq_mask_x, seq_len_x=seq_len_x,
        y_in=y_in, y_out=y_out, seq_mask_y=seq_mask_y, seq_len_y=seq_len_y,
        step=step, KL_weight=KL_weight,
        tracker=tracker, writer=writer,
        hparams=hparams, title=title
    )
    loss = loss + regularizer
    loss.backward()
    take_optimizer_step(optimizers['gen'], model.generative_parameters(), hparams.max_gradient_norm)
    take_optimizer_step(optimizers['inf_z'], model.inference_parameters(), hparams.max_gradient_norm)



def train(model,
          optimizers, lr_schedulers,
          data_xy,
          val_data, vocab_src, vocab_tgt,
          device, out_dir, validate, hparams,data_x, data_y):
    """
    :param train_step: function that performs a single training step and returns
                       training loss. Takes as inputs: model, x_in, x_out,
                       seq_mask_x, seq_len_x, y_in, y_out, seq_mask_y,
                       seq_len_y, hparams, step.
    :param validate: function that performs validation and returns validation
                     BLEU, used for model selection. Takes as inputs: model,
                     val_data, vocab, device, hparams, step, summary_writer.
                     summary_writer can be None if no summaries should be made.
                     This function should perform all evaluation, write
                     summaries and write any validation metrics to the
                     standard out.
    """

    # Create a dataloader that buckets the batches.
    dl_xy = DataLoader(
        dataset=data_xy,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=4)
    bucketing_dl_xy = BucketingParallelDataLoader(dl_xy)  # n=1 to synchronise with dl_yx

    if data_x:
        dl_x = DataLoader(
            dataset=data_x, batch_size=hparams.batch_size,
            shuffle=True, num_workers=4)
        bucketing_dl_x = BucketingTextDataLoader(dl_x)
        cycle_iterate_dl_x = cycle(bucketing_dl_x)

    if data_y:
        dl_y = DataLoader(
            dataset=data_y, batch_size=hparams.batch_size,
            shuffle=True, num_workers=4)
        bucketing_dl_y = BucketingTextDataLoader(dl_y)
        cycle_iterate_dl_y = cycle(bucketing_dl_y)

    # Keep track of some stuff in TensorBoard.
    summary_writer = SummaryWriter(log_dir=str(out_dir))

    # Time, epoch, steps
    tokens_start = time.time()
    epoch_num = 1
    step_counter = StepCounter()

    # Fix curriculum and iterators
    curriculum = hparams.curriculum.split()
    if 'x' in curriculum and hparams.mono_src is None:
        raise ValueError("You have scheduled source monolingual data but have not specified --mono_src")

    if 'y' in curriculum and hparams.mono_tgt is None:
        raise ValueError("You have scheduled target monolingual data but have not specified --mono_tgt")

    # count number of steps in a epoch
    nb_bilingual_batches = count_number_of_batches(iter(bucketing_dl_xy))
    cycle_iterate_dl_xy = cycle(bucketing_dl_xy)
    cycle_curriculum = cycle(curriculum)

    # Manage checkpoints (depends on training phase)
    ckpt = CheckPoint(model_dir=out_dir, metrics=['bleu', 'likelihood'])

    # Define the evaluation function.
    def run_evaluation(step,writer=summary_writer):
        # Perform model validation, keep track of validation BLEU for model
        # selection.
        model.eval()
        metrics = validate(model, val_data, vocab_src, vocab_tgt, device,
                            hparams, step, summary_writer=writer)

        # Update the learning rate scheduler.
        lr_scheduler_step(lr_schedulers, hparams, val_score=metrics[hparams.criterion])

        ckpt.update(
            epoch_num, step, {f"{hparams.src}-{hparams.tgt}": model},
            # we save with respect to BLEU and likelihood
            bleu=metrics['bleu'], likelihood=metrics['likelihood']
        )

    # Some statistics for training
    tracker_xy = Tracker(hparams.print_every)
    tracker_x, tracker_y = Tracker(hparams.print_every), Tracker(hparams.print_every)

    # Start the training loop.
    KL_weight = 1.
    while (epoch_num <= hparams.num_epochs) or (ckpt.no_improvement(hparams.criterion) < hparams.patience):
        waiting = nb_bilingual_batches
        while waiting:
            batch_type = next(cycle_curriculum)
            model.train()

            # Do linear annealing of the KL over KL_annealing_steps if set.
            if hparams.KL_annealing_steps > 0:
                KL_weight = min(1., (1.0 / hparams.KL_annealing_steps) * step_counter.step('xy'))

            if batch_type == 'y':  # target monolingual batch
                if epoch_num > hparams.warmup:
                    backtranslated_examples = dict()
                    sentences_y = next(cycle_iterate_dl_y)
                    y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in = create_noisy_batch(
                        sentences_y, vocab_tgt, device, word_dropout=0.)  # TODO: should we use word dropout?
                    aevnmt_monolingual_step(
                        model_xy, model_y, vocab_src,
                        y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
                        hparams,
                        step=step_counter.step(),
                        device=device,
                        optimizers_xy=optimizers_xy,
                        optimizers_yx=optimizers_yx,  # for mono VAE loss
                        KL_weight=KL_weight,
                        tracker=tracker_y,
                        reward_stats=r_stats_y,
                        gather_examples=backtranslated_examples if (step_counter.step(
                            'y') + 1) % hparams.print_every == 0 else None,
                        writer=summary_writer if step_counter.step('y') % hparams.print_every == 0 else None,
                        title='mono_tgt')

                    step_counter.count('y')

                    if hparams.inspect_backtranslation and step_counter.step('y') % hparams.print_every == 0:
                        print("Back-translation (target-to-source):")
                        ridx = np.random.choice(len(sentences_y))
                        print(f" - Input: {sentences_y[ridx]}")
                        print(f" - Sample: {backtranslated_examples['sampled_x'][ridx]}")
                        if 'greedy_x' in backtranslated_examples:
                            print(f" - Greedy: {backtranslated_examples['greedy_x'][ridx]}")

            elif batch_type == 'x':  # source monolingual batch
                sentences_x = next(cycle_iterate_dl_x)
                x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in = create_noisy_batch(
                    sentences_x, vocab_src, device, word_dropout=hparams.word_dropout)  # TODO: should we use word dropout?
                senvae_monolingual_step_x(
                    model=model, inputs=x_in, noisy_inputs=noisy_x_in, targets=x_out,
                    seq_mask=seq_mask_x, seq_len=seq_len_x,
                    step=step_counter.step(),
                    optimizers=optimizers, KL_weight=KL_weight,
                    hparams=hparams,
                    tracker=tracker_x,
                    writer=summary_writer if step_counter.step('x') % hparams.print_every == 0 else None,
                    title="mono_src/SenVAE"
                )
                step_counter.count('x')
            elif batch_type == 'xy':  # bilingual batch

                sentences_x, sentences_y = next(cycle_iterate_dl_xy)
                x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in = create_noisy_batch(
                    sentences_x, vocab_src, device, word_dropout=hparams.word_dropout)

                y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in = create_noisy_batch(
                    sentences_y, vocab_tgt, device, word_dropout=hparams.word_dropout)

                aevnmt_bilingual_step_xy(model=model, hparams=hparams,
                x_in=noisy_x_in, x_out=x_out, seq_mask_x=seq_mask_x, seq_len_x=seq_len_x,
                y_in=noisy_y_in, y_out=y_out, seq_mask_y=seq_mask_y, seq_len_y=seq_len_y,
                step=step_counter.step(), optimizers=optimizers,lr_schedulers=lr_schedulers,
                KL_weight=KL_weight,
                tracker=tracker_xy,
                writer=summary_writer if step_counter.step(
                    'xy') % hparams.print_every == 0 else None,
                title="bilingual/xy")

                # Run evaluation every evaluate_every steps if set (always after a bilingual batch)
                if hparams.evaluate_every > 0 and step_counter.step('xy') % hparams.evaluate_every == 0:
                    run_evaluation(step_counter.step())

                # Print training stats every now and again.
                if step_counter.step('xy') % hparams.print_every == 0:
                    elapsed = time.time() - tokens_start
                    num_tokens = tracker_x.sum('num_tokens') + tracker_y.sum('num_tokens')
                    num_tokens += tracker_xy.sum('num_tokens')
                    tokens_per_sec = num_tokens / elapsed if step_counter.step() != 0 else 0
                    print(f"({epoch_num}) step {step_counter.step()} "
                          f"(xy: {step_counter.step('xy')} "
                          f"x: {step_counter.step('x')} "
                          f"y: {step_counter.step('y')}) "
                          f"AEVNMT(x,y) = {tracker_xy.avg('ELBO', 'num_sentences'):,.2f} -- "
                          f"AEVNTM(x) = {tracker_x.avg('ELBO', 'num_sentences'):,.2f} -- "
                          f"AEVNMT(y) = {tracker_y.avg('ELBO', 'num_sentences'):,.2f} -- "
                          f"SenVAE(x) = {tracker_x.avg('SenVAE/ELBO', 'num_sentences'):,.2f} -- "
                          f"SenVAE(y) = {tracker_y.avg('SenVAE/ELBO', 'num_sentences'):,.2f} -- "
                          f"{tokens_per_sec:,.0f} tokens/s -- "
                          f"{elapsed:,.0f} s -- ")

                    tokens_start = time.time()

                step_counter.count('xy')
                waiting -= 1

        print(f"Finished epoch {epoch_num}")

        # If evaluate_every is not set, we evaluate after every epoch.
        if hparams.evaluate_every <= 0:
            run_evaluation(step_counter.step())

        epoch_num += 1

    print(f"Finished training.")
    summary_writer.close()

    # Load the best model and run validation again, make sure to not write
    # summaries.
    model_xy.load_state_dict(torch.load(out_dir / f"model.{hparams.src}-{hparams.tgt}.pt"))
    model_yx.load_state_dict(torch.load(out_dir / f"model.{hparams.tgt}-{hparams.src}.pt"))
    print(f"Loaded best model found at step {best_step} (epoch {best_epoch}).")
    run_evaluation(step_counter.step(), writer=None)


def main():

    # Load and print hyperparameters.
    hparams = Hyperparameters()
    print("\n==== Hyperparameters")
    hparams.print_values()

    # Create the output directory and save hparams
    out_dir = Path(hparams.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hparams.save(out_dir / "hparams")
    print("\n==== Output")
    print(f"Created output directory at {hparams.output_dir}")

    # Load/construct and possibly save vocabulary
    vocab_src, vocab_tgt = load_vocabularies(hparams)
    if hparams.vocab_prefix is None:
        vocab_src.save(out_dir / f"vocab.{hparams.src}")
        vocab_tgt.save(out_dir / f"vocab.{hparams.tgt}")
        hparams.vocab_prefix = out_dir / "vocab"

    # Load datasets (possibly construct memory maps)
    train_data, val_data, opt_data = load_data(hparams, vocab_src, vocab_tgt, use_memmap=hparams.use_memmap)

    print("\n==== Data")
    print(f"Training data: {len(train_data):,} bilingual sentence pairs")
    print('', ' ||| '.join(train_data[np.random.choice(len(train_data))]))
    print(f"Validation data: {len(val_data):,} bilingual sentence pairs")
    print('', ' ||| '.join(train_data[np.random.choice(len(val_data))]))
    if 'mono_src' in opt_data:
        print(f"Source monolingual data: {len(opt_data['mono_src']):,} sentences")
        print(f" {opt_data['mono_src'][np.random.choice(len(opt_data['mono_src']))]}")
    if 'mono_tgt' in opt_data:
        print(f"Target monolingual data: {len(opt_data['mono_tgt']):,} sentences")
        print(f" {opt_data['mono_tgt'][np.random.choice(len(opt_data['mono_tgt']))]}")

    # Create the language model and load it onto the GPU if set to do so.
    model, train_fn, validate_fn, _,_ = create_model(hparams, vocab_src, vocab_tgt)

    optimizers, lr_schedulers = construct_optimizers(
        hparams,
        gen_parameters=model.generative_parameters(),
        inf_z_parameters=model.inference_parameters())
    device = torch.device("cuda:0") if hparams.use_gpu else torch.device("cpu")
    model = model.to(device)

    # Print information about the model.
    print("\n==== Model")
    print("Short summary:")
    print(model)
    print("\nAll parameters:")
    for name, param in model.named_parameters():
        print(f"{name} -- {param.size()}")


    # Initialize the model parameters, or load a checkpoint.
    if hparams.model_checkpoint is None:
        print("\nInitializing parameters...")
        initialize_model(model, vocab_tgt[PAD_TOKEN], hparams.cell_type,
                         hparams.emb_init_scale, verbose=True)
    else:
        print(f"\nRestoring model parameters from {hparams.model_checkpoint}...")
        model.load_state_dict(torch.load(hparams.model_checkpoint))

    # Create the output directories.
    out_dir = Path(hparams.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hparams.vocab_prefix is None:
        vocab_src.save(out_dir / f"vocab.{hparams.src}")
        vocab_tgt.save(out_dir / f"vocab.{hparams.tgt}")
        hparams.vocab_prefix = out_dir / "vocab"
    hparams.save(out_dir / "hparams")
    print("\n==== Output")
    print(f"Created output directory at {hparams.output_dir}")

    # Train the model.
    print("\n==== Starting training")
    print(f"Using device: {device}\n")
    train(model, optimizers, lr_schedulers, train_data, val_data, vocab_src,
          vocab_tgt, device, out_dir, validate_fn, hparams,data_x=opt_data.get('mono_src', None),data_y=opt_data.get('mono_tgt', None))


if __name__ == "__main__":
    main()
