import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import aevnmt.aevnmt_helper as aevnmt_helper
from aevnmt.components import sampling_decode
from aevnmt.data import batch_to_sentences, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ParallelDatasetFlippedView

from itertools import cycle, chain

from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from pathlib import Path
from tensorboardX import SummaryWriter

from aevnmt.train_utils import load_data, load_vocabularies_senvae
from aevnmt.train_utils import StepCounter, CheckPoint
from aevnmt.opt_utils import construct_optimizers, lr_scheduler_step, take_optimizer_step, RequiresGradSwitch, get_optimizer
from aevnmt.hparams import Hyperparameters
from aevnmt.data import BucketingParallelDataLoader, BucketingTextDataLoader
from aevnmt.data import PAD_TOKEN
from aevnmt.data.utils import create_noisy_batch
from aevnmt.models import initialize_model

from collections import defaultdict, deque
import numpy as np

from aevnmt.data import Vocabulary, ParallelDataset, TextDataset, remove_subword_tokens


from aevnmt.dist import get_named_params

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
        model, hparams, inputs, noisy_inputs, targets, seq_len,seq_mask,
        inputs_shuf, noisy_inputs_shuf, targets_shuf, seq_len_shuf,seq_mask_shuf,
        qz, z, KL_weight, step,
        state=dict(), writer=None, title="SenVAE", disable_main_loss=False,disable_side_losses=False, disable_kl=False):

    tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = model(noisy_inputs, seq_mask, seq_len, None,
    noisy_inputs_shuf,seq_mask_shuf,seq_len_shuf,
    None,None,None,
    z)

    y_out=None
    y_shuf_out=None
    loss = model.loss(tm_likelihood, lm_likelihood, y_out, targets,y_shuf_out,targets_shuf, qz,
                      free_nats=hparams.KL_free_nats,
                      KL_weight=KL_weight * ( 0 if disable_kl else 1 ),
                      reduction="mean",
                      aux_lm_likelihoods=aux_lm_likelihoods,
                      aux_tm_likelihoods=aux_tm_likelihoods, disable_main_loss=disable_main_loss,disable_side_losses=disable_side_losses)


    if writer:
        writer.add_scalar('%s/KL' % title, loss['raw_KL'].mean(), step)
        writer.add_scalar('%s/LL' % title, (loss['lm/main']).mean(), step)
        writer.add_scalar('%s/ELBO' % title, loss['ELBO'].mean(), step)
        if 'sideELBO' in loss:
            writer.add_scalar('%s/sideELBO' % title, loss['sideELBO'].mean(), step)
        if 'lag_side_loss' in loss:
            writer.add_scalar('%s/lag_side_loss' % title, loss['lag_side_loss'].mean(), step)

    return loss


def senvae_monolingual_step_x(
        model,
        inputs, noisy_inputs, targets, seq_mask, seq_len,
        inputs_shuf, noisy_inputs_shuf, targets_shuf, seq_mask_shuf, seq_len_shuf,
        step, optimizers, KL_weight,
        hparams, tracker, writer=None, title='SenVAE',disable_main_loss=False,disable_side_losses=False,disable_kl=False,disconnect_inference_network=False):

    # Infer q(z|x)
    qz = model.approximate_posterior(inputs, seq_mask, seq_len,None,None,None)
    # [B, dz]
    z = qz.rsample()
    # [B]

    if disconnect_inference_network:
        z_in=z.detach()
    else:
        z_in=z

    if writer:
        writer.add_histogram("posterior-x/z", z_in, step)
        for param_name, param_value in get_named_params(qz):
            writer.add_histogram("posterior-x/%s" % param_name, param_value, step)

    mono_vae_terms = mono_vae_loss(
        model=model, hparams=hparams,
        inputs=inputs, noisy_inputs=noisy_inputs, targets=targets, seq_len=seq_len,seq_mask=seq_mask,
        inputs_shuf=inputs_shuf, noisy_inputs_shuf=noisy_inputs_shuf, targets_shuf=targets_shuf, seq_mask_shuf=seq_mask_shuf, seq_len_shuf=seq_len_shuf,
        qz=qz, z=z_in,
        KL_weight=KL_weight, step=step,
        writer=writer, title=title, disable_main_loss=disable_main_loss, disable_side_losses=disable_side_losses,disable_kl=disable_kl
    )
    negative_elbo = mono_vae_terms['loss']
    negative_elbo.backward()

    if hparams.max_gradient_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                 max_norm=hparams.max_gradient_norm,
                                 norm_type=float("inf"))

    optimizers['gen'].step()
    optimizers['inf_z'].step()

    optimizers['gen'].zero_grad()
    optimizers['inf_z'].zero_grad()

    if hparams.mdr:
        optimizers['mdr'].zero_grad()
        mono_vae_terms['mdr_loss'].backward()
        optimizers['mdr'].step()
        tracker.update('MRD/loss', mono_vae_terms['mdr_loss'].sum().item())

    if hparams.lag_side is not None:
        optimizers['lag_side'].zero_grad()
        mono_vae_terms['lag_side_loss'].backward()
        optimizers['lag_side'].step()
        tracker.update('LagSide/loss', mono_vae_terms['lag_side_loss'].sum().item())
        tracker.update('LagSide/difference', mono_vae_terms['lag_difference'].sum().item())

    # Update statistics.
    ELBO = mono_vae_terms['ELBO']
    sideELBO = mono_vae_terms['sideELBO']
    tracker.update('SenVAE/ELBO', ELBO.sum().item())
    tracker.update('SenVAE/sideELBO', sideELBO.sum().item())
    tracker.update('SenVAE/sideLoss', sideELBO.sum().item())
    tracker.update('num_tokens', seq_len.sum().item())
    tracker.update('num_sentences', inputs.size(0))


def train(model,
          optimizers, lr_schedulers,
          data_x,
          val_data, test_data, vocab_src,
          device, out_dir, validate, hparams):
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
    dl_x = DataLoader(
        dataset=data_x,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=4)
    bucketing_dl_x = BucketingTextDataLoader(dl_x)  # n=1 to synchronise with dl_yx

    if hparams.disable_tensorboard:
        summary_writer =None
    else:
        # Keep track of some stuff in TensorBoard.
        summary_writer = SummaryWriter(log_dir=str(out_dir))

    # Time, epoch, steps
    tokens_start = time.time()
    epoch_num = 1
    step_counter = StepCounter()

    # count number of steps in a epoch
    nb_bilingual_batches = count_number_of_batches(iter(bucketing_dl_x))
    cycle_iterate_dl_x = cycle(bucketing_dl_x)

    # Manage checkpoints (depends on training phase)
    ckpt = CheckPoint(model_dir=out_dir/"model", metrics=['bleu', 'likelihood', 'side_likelihood'])

    only_side_losses_phase=False
    if hparams.side_losses_warmup_convergence_patience > 0:
        only_side_losses_phase=True
    side_losses_vals=[]


    # Define the evaluation function.
    def run_evaluation(step,only_side_losses_phase,data,writer=summary_writer,num_importance_samples=10):
        # Perform model validation, keep track of validation BLEU for model
        # selection.
        model.eval()
        metrics = validate(model, data, vocab_src, None, device,
                            hparams, step, summary_writer=writer,num_importance_samples=num_importance_samples)


        if (not epoch_num <= hparams.side_losses_warmup) and not only_side_losses_phase:
            # Update the learning rate scheduler.
            cooldown=lr_scheduler_step(lr_schedulers, hparams, val_score=metrics[hparams.criterion])

            ckpt.update(
                epoch_num, step, {f"{hparams.src}-{hparams.tgt}": model},
                # we save with respect to BLEU and likelihood
                bleu=metrics['bleu'], likelihood=metrics['likelihood'] , side_likelihood=metrics['side_likelihood']
            )
            if cooldown:
                ckpt.cooldown_happened()


        side_losses_vals.append(metrics['side_NLL'])
        if hparams.side_losses_warmup_convergence_patience > 0 and only_side_losses_phase and min(side_losses_vals) not in side_losses_vals[-hparams.side_losses_warmup_convergence_patience:]:
            if hparams.reset_main_decoder_after_warmup:
                #reset main decoder
                initialize_model(model.language_model, vocab_src[PAD_TOKEN], hparams.cell_type,
                                 hparams.emb_init_scale, verbose=True)
                optimizers["gen"] =get_optimizer(
                        hparams.gen_optimizer,
                        model.generative_parameters(),
                        hparams.gen_lr,
                        hparams.gen_l2_weight)
            only_side_losses_phase=False

        return only_side_losses_phase

    # Some statistics for training
    tracker_x = Tracker(hparams.print_every)

    shuffle_dict_sl=dict()
    # Start the training loop.
    KL_weight = 1.
    while (epoch_num <= hparams.num_epochs) or (ckpt.no_improvement(hparams.criterion) < hparams.patience and ckpt.cooldowns < hparams.cooldown_patience ):
        waiting = nb_bilingual_batches
        while waiting:
            batch_type = 'x'
            model.train()

            # Do linear annealing of the KL over KL_annealing_steps if set.
            if hparams.KL_annealing_steps > 0:
                KL_weight = min(1., (1.0 / hparams.KL_annealing_steps) * step_counter.step('x'))

            if batch_type == 'x':  # source monolingual batch
                sentences_x = next(cycle_iterate_dl_x)
                x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in = create_noisy_batch(
                    sentences_x, vocab_src, device, word_dropout=hparams.word_dropout)  # TODO: should we use word dropout?

                if 'shuffled' in model.aux_lms:
                    x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in=create_noisy_batch(
                        sentences_x, vocab_src, device,
                        word_dropout=hparams.word_dropout,shuffle_toks=True,full_words_shuf=hparams.shuffle_lm_keep_bpe,shuffle_dict=shuffle_dict_sl if hparams.shuffle_lm_keep_epochs else None)
                else:
                    x_shuf_in=x_shuf_out=seq_mask_x_shuf=seq_len_x_shuf=noisy_x_shuf_in=None

                senvae_monolingual_step_x(
                    model=model, inputs=x_in, noisy_inputs=noisy_x_in, targets=x_out,seq_mask=seq_mask_x, seq_len=seq_len_x,
                    inputs_shuf=x_shuf_in, noisy_inputs_shuf=noisy_x_shuf_in, targets_shuf=x_shuf_out, seq_mask_shuf=seq_mask_x_shuf, seq_len_shuf=seq_len_x_shuf,
                    step=step_counter.step(),
                    optimizers=optimizers, KL_weight=KL_weight,
                    hparams=hparams,
                    tracker=tracker_x,
                    writer=summary_writer if step_counter.step('x') % hparams.print_every == 0 else None,
                    title="mono_src/SenVAE",
                    disable_main_loss= ( ( (epoch_num <= hparams.side_losses_warmup) or only_side_losses_phase) and not hparams.keep_main_loss_during_warmup ) or hparams.disable_main_loss,
                    disable_side_losses =(( (epoch_num > hparams.side_losses_warmup and  hparams.side_losses_warmup > 0 ) or (hparams.side_losses_warmup_convergence_patience > 0 and not only_side_losses_phase)) and hparams.disable_side_losses_after_warmup),
                    disable_kl =(( (epoch_num > hparams.side_losses_warmup and  hparams.side_losses_warmup > 0 ) or (hparams.side_losses_warmup_convergence_patience > 0 and not only_side_losses_phase)) and hparams.disable_KL_after_warmup),
                    disconnect_inference_network =(( (epoch_num > hparams.side_losses_warmup and  hparams.side_losses_warmup > 0 ) or (hparams.side_losses_warmup_convergence_patience > 0 and not only_side_losses_phase)) and hparams.disconnect_inference_network_after_warmup)


                )
                step_counter.count('x')

                # Run evaluation every evaluate_every steps if set (always after a bilingual batch)
                if hparams.evaluate_every > 0 and step_counter.step('x') % hparams.evaluate_every == 0:
                    only_side_losses_phase=run_evaluation(step_counter.step(),only_side_losses_phase,val_data)

                # Print training stats every now and again.
                if step_counter.step('x') % hparams.print_every == 0:

                    elapsed = time.time() - tokens_start
                    num_tokens = tracker_x.sum('num_tokens')
                    tokens_per_sec = num_tokens / elapsed if step_counter.step() != 0 else 0
                    bias=0.0
                    u=0.0
                    if model.lag_side is not None:
                        bias=model.lag_side[1].bias.item()
                        with torch.no_grad():
                            u=model.lag_side(torch.zeros(1,device=x_in.device)).item()
                    print(f"({epoch_num}) step {step_counter.step()} "
                          f"x: {step_counter.step('x')} "
                          f"SenVAE(x) = {tracker_x.avg('SenVAE/ELBO', 'num_sentences'):,.2f} -- "
                          f"side(x) = {tracker_x.avg('SenVAE/sideELBO', 'num_sentences'):,.2f} -- "
                          f"per_token_side_loss(x) = {tracker_x.avg('SenVAE/sideLoss', 'num_tokens'):,.2f} -- "
                          f"lag_side(x) = {tracker_x.mean('LagSide/loss'):,.2f} bias= {bias:,.2f} u={u:,.2f}  -- "
                          f"lag_diff(x) = {tracker_x.mean('LagSide/difference'):,.2f} -- "
                          f"{tokens_per_sec:,.0f} tokens/s -- "
                          f"{elapsed:,.0f} s -- ")

                    tokens_start = time.time()

                waiting -= 1

        print(f"Finished epoch {epoch_num}")

        # If evaluate_every is not set, we evaluate after every epoch.
        if hparams.evaluate_every <= 0:
            only_side_losses_phase=run_evaluation(step_counter.step(),only_side_losses_phase,val_data)

        epoch_num += 1

    print(f"Finished training.")
    if summary_writer is not None:
        summary_writer.close()

    # Load the best model and run validation again, make sure to not write
    # summaries.
    best_model_info = ckpt.load_best({f"{hparams.src}-{hparams.tgt}": model}, hparams.criterion)
    print(f"Loaded best model (wrt {hparams.criterion}) found at step {best_model_info['step']} (epoch {best_model_info['epoch']}).")
    run_evaluation(step_counter.step(),only_side_losses_phase, val_data,writer=None)

    if test_data is not None:
        print(f"Test set evaluation.")
        run_evaluation(step_counter.step(),only_side_losses_phase, test_data,writer=None,num_importance_samples=1000)


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
    vocab_src = load_vocabularies_senvae(hparams)
    if hparams.vocab_prefix is None:
        vocab_src.save(out_dir / f"vocab.{hparams.src}")
        hparams.vocab_prefix = out_dir / "vocab"

    # Load datasets (possibly construct memory maps)
    assert not hparams.use_memmap, "use_memmap not supported"
    train_data=TextDataset(hparams.mono_src, max_length=hparams.max_sentence_length)
    val_data= TextDataset(f"{hparams.validation_prefix}.{hparams.src}", max_length=-1)
    if hparams.test_prefix is not None:
        test_data= TextDataset(f"{hparams.test_prefix}.{hparams.src}", max_length=-1)
    else:
        test_data=None

    print("\n==== Data")
    print(f"Training data: {len(train_data):,} sentences")
    print('', train_data[np.random.choice(len(train_data))] )
    print(f"Validation data: {len(val_data):,} sentences")
    print('', train_data[np.random.choice(len(val_data))])

    # Create the language model and load it onto the GPU if set to do so.
    model, train_fn, validate_fn, _,_ = create_model(hparams, vocab_src, None)

    optimizers, lr_schedulers = construct_optimizers(
        hparams,
        gen_parameters=model.generative_parameters(),
        inf_z_parameters=model.inference_parameters(),
        mdr_parameters=model.mdr_parameters(),
        lag_side_parameters=model.lag_side_parameters())
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
        initialize_model(model, vocab_src[PAD_TOKEN], hparams.cell_type,
                         hparams.emb_init_scale, verbose=True)
    else:
        print(f"\nRestoring model parameters from {hparams.model_checkpoint}...")
        model.load_state_dict(torch.load(hparams.model_checkpoint))

    # Create the output directories.
    out_dir = Path(hparams.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hparams.vocab_prefix is None:
        vocab_src.save(out_dir / f"vocab.{hparams.src}")
        hparams.vocab_prefix = out_dir / "vocab"
    hparams.save(out_dir / "hparams")
    print("\n==== Output")
    print(f"Created output directory at {hparams.output_dir}")

    # Train the model.
    print("\n==== Starting training")
    print(f"Using device: {device}\n")
    train(model, optimizers, lr_schedulers, train_data, val_data, test_data, vocab_src,
           device, out_dir, validate_fn, hparams)


if __name__ == "__main__":
    main()
