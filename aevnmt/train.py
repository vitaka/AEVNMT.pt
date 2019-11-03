import torch
import torch.nn as nn
import time

import numpy as np

import aevnmt.nmt_helper as nmt_helper
import aevnmt.aevnmt_helper as aevnmt_helper
import aevnmt.vae_helper as vae_helper

from torch.utils.data import DataLoader
from torch import autograd
from pathlib import Path
from tensorboardX import SummaryWriter

from aevnmt.train_utils import load_data, load_vocabularies, gradient_norm
from aevnmt.train_utils import CheckPoint, model_parameter_count
from aevnmt.hparams import Hyperparameters
from aevnmt.data import BucketingParallelDataLoader
from aevnmt.data import PAD_TOKEN
from aevnmt.data.utils import create_noisy_batch
from aevnmt.models import initialize_model

from aevnmt.opt_utils import construct_optimizers, lr_scheduler_step


def create_model(hparams, vocab_src, vocab_tgt):
    if hparams.model_type == "cond_nmt":
        model = nmt_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = nmt_helper.train_step
        validate_fn = nmt_helper.validate
        translate_fn = nmt_helper.translate
    elif hparams.model_type == "aevnmt":
        model = aevnmt_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = aevnmt_helper.train_step
        validate_fn = aevnmt_helper.validate
        translate_fn = aevnmt_helper.translate
    elif hparams.model_type == "vae":
        model = vae_helper.create_model(hparams, vocab_src, vocab_tgt)
        train_fn = vae_helper.train_step
        validate_fn = vae_helper.validate
        translate_fn = vae_helper.translate
    else:
        raise Exception(f"Unknown model_type: {hparams.model_type}")

    return model, train_fn, validate_fn, translate_fn


def train(model, optimizers, lr_schedulers, training_data, val_data, vocab_src,
          vocab_tgt, device, out_dir, train_step, validate, hparams):
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
    dl = DataLoader(training_data, batch_size=hparams.batch_size,
                    shuffle=True, num_workers=4)
    bucketing_dl = BucketingParallelDataLoader(dl,add_reverse=hparams.reverse_lm or hparams.shuffle_lm)

    # Save the best model based on development BLEU.
    ckpt = CheckPoint(model_dir=out_dir/"model", metrics=['bleu', 'likelihood'])

    # Keep track of some stuff in TensorBoard.
    summary_writer = SummaryWriter(log_dir=str(out_dir))

    # Define training statistics to keep track of.
    tokens_start = time.time()
    num_tokens = 0
    total_train_loss = 0.
    total_train_loss_masked_lm = 0.
    num_sentences = 0
    step = 0
    epoch_num = 1

    # Define the evaluation function.
    def run_evaluation():
        # Perform model validation, keep track of validation BLEU for model
        # selection.
        model.eval()
        metrics = validate(model, val_data, vocab_src, vocab_tgt, device,
                            hparams, step, summary_writer=summary_writer)

        # Update the learning rate scheduler.
        lr_scheduler_step(lr_schedulers, hparams, val_score=metrics[hparams.criterion])

        ckpt.update(
            epoch_num, step, {f"{hparams.src}-{hparams.tgt}": model},
            # we save with respect to BLEU and likelihood
            bleu=metrics['bleu'], likelihood= metrics['main_likelihood'] if hparams.multi_task_stop_main else metrics['likelihood']
        )

    # Start the training loop.
    while (epoch_num <= hparams.num_epochs) or (ckpt.no_improvement(hparams.criterion) < hparams.patience):

        # Train for 1 epoch.
        for sentences_tuple in bucketing_dl:
            if hparams.reverse_lm or hparams.shuffle_lm:
                sentences_x, sentences_y, sentences_x_rev, sentences_y_rev, sentences_x_shuf, sentences_y_shuf =sentences_tuple
            else:
                sentences_x, sentences_y =sentences_tuple
                sentences_x_rev =sentences_y_rev =sentences_x_shuf= sentences_y_shuf=None
            model.train()

            # Perform a forward pass through the model
            x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in = create_noisy_batch(
                sentences_x, vocab_src, device,
                word_dropout=hparams.word_dropout)
            y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in = create_noisy_batch(
                sentences_y, vocab_tgt, device,
                word_dropout=hparams.word_dropout)

            if sentences_x_rev is not None:
                x_rev_in, x_rev_out, seq_mask_x_rev, seq_len_x_rev, noisy_x_rev_in = create_noisy_batch(
                    sentences_x_rev, vocab_src, device,
                    word_dropout=hparams.word_dropout)
                y_rev_in, y_rev_out, seq_mask_y_rev, seq_len_y_rev, noisy_y_rev_in = create_noisy_batch(
                    sentences_y_rev, vocab_tgt, device,
                    word_dropout=hparams.word_dropout)
            else:
                x_rev_in= x_rev_out= seq_mask_x_rev= seq_len_x_rev= noisy_x_rev_in =None
                y_rev_in= y_rev_out= seq_mask_y_rev= seq_len_y_rev= noisy_y_rev_in=None

            if sentences_x_shuf is not None:
                x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in = create_noisy_batch(
                    sentences_x_shuf, vocab_src, device,
                    word_dropout=hparams.word_dropout)
                y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf, noisy_y_shuf_in = create_noisy_batch(
                    sentences_y_shuf, vocab_tgt, device,
                    word_dropout=hparams.word_dropout)
            else:
                x_shuf_in= x_shuf_out= seq_mask_x_shuf= seq_len_x_shuf= noisy_x_shuf_in =None
                y_shuf_in= y_shuf_out= seq_mask_y_shuf= seq_len_y_shuf= noisy_y_shuf_in=None


            x_to_y=False
            y_to_x=False
            if np.random.random() < hparams.cross_language_dropout:
                if np.random.random() <= 0.5:
                    x_to_y=True
                else:
                    y_to_x=True

            #if step==15:
            #    import pdb; pdb.set_trace()

            #with autograd.detect_anomaly():
            train_result = train_step(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in,
                              y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in, x_rev_in, x_rev_out, seq_mask_x_rev, seq_len_x_rev, noisy_x_rev_in, y_rev_in, y_rev_out, seq_mask_y_rev, seq_len_y_rev, noisy_y_rev_in,
                              x_shuf_in, x_shuf_out, seq_mask_x_shuf, seq_len_x_shuf, noisy_x_shuf_in, y_shuf_in, y_shuf_out, seq_mask_y_shuf, seq_len_y_shuf, noisy_y_shuf_in, hparams, step, add_qz_scale=0.00000001 if step<=hparams.avoid_zero_scale_during else 0.0, x_to_y=x_to_y,y_to_x=y_to_x)
            loss=train_result["loss"]
            # Backpropagate and update gradients.
            loss.backward()

            if hparams.max_gradient_norm > 0:
                # TODO: do we need separate norms?
                nn.utils.clip_grad_norm_(model.parameters(),
                                         hparams.max_gradient_norm)
            optimizers["gen"].step()
            if "inf_z" in optimizers: optimizers["inf_z"].step()

            # Update statistics.
            num_tokens += (seq_len_x.sum() + seq_len_y.sum()).item() *( 1 + (1 if hparams.reverse_lm else 0) + (1 if hparams.shuffle_lm else 0) )

            num_sentences += x_in.size(0)
            total_train_loss += loss.item() * x_in.size(0)
            total_train_loss_masked_lm +=train_result["masked_lm_log_likelihood"].sum() if not isinstance(train_result["masked_lm_log_likelihood"],float) != 0 else train_result["masked_lm_log_likelihood"]

            # Print training stats every now and again.
            if step % hparams.print_every == 0:

                #print (model.bow_output_layer.weight)
                elapsed = time.time() - tokens_start
                tokens_per_sec = num_tokens / elapsed if step != 0 else 0


                if hparams.print_gradient_norms:
                    for name, p in model.named_parameters():
                        param_norm = p.grad.data.norm(2)
                        print("{}: {}".format(name,param_norm))

                grad_norm = gradient_norm(model)
                print(f"({epoch_num}) step {step}: "
                       f"training loss = {total_train_loss/num_sentences:,.2f} -- "
                       f"training loss masked LM = {total_train_loss_masked_lm/num_sentences:,.2f} -- "
                       f"{tokens_per_sec:,.0f} tokens/s -- "
                       f"gradient norm = {grad_norm:.2f}")
                summary_writer.add_scalar("train/loss",
                                          total_train_loss/num_sentences, step)
                summary_writer.add_histogram("train/z",
                                          train_result["z"], step)
                for k in train_result:
                    if k != "z" and k!= "loss":
                        summary_writer.add_scalar("train/auto-"+k,
                                              (train_result[k].sum() if (not isinstance(train_result[k], float) and not isinstance(train_result[k], int)) else train_result[k])/num_sentences, step)

                num_tokens = 0
                tokens_start = time.time()
                total_train_loss = 0.
                total_train_loss_masked_lm = 0.
                num_sentences = 0

            # Zero the gradient buffer.
            optimizers["gen"].zero_grad()
            if "inf_z" in optimizers: optimizers["inf_z"].zero_grad()

            # Update the learning rate scheduler if needed.
            lr_scheduler_step(lr_schedulers, hparams)

            # Run evaluation every evaluate_every steps if set.
            if hparams.evaluate_every > 0 and step > 0 and step % hparams.evaluate_every == 0:
                run_evaluation()

            step += 1

        print(f"Finished epoch {epoch_num}")

        # If evaluate_every is not set, we evaluate after every epoch.
        if hparams.evaluate_every <= 0:
            run_evaluation()

        epoch_num += 1

    print(f"Finished training.")
    summary_writer.close()

    # Load the best model and run validation again, make sure to not write
    # summaries.
    best_model_info = ckpt.load_best({f"{hparams.src}-{hparams.tgt}": model}, hparams.criterion)
    print(f"Loaded best model (wrt {hparams.criterion}) found at step {best_model_info['step']} (epoch {best_model_info['epoch']}).")
    model.eval()
    validate(model, val_data, vocab_src, vocab_tgt, device, hparams, step,
             summary_writer=None)


def main():

    # Load and print hyperparameters.
    hparams = Hyperparameters()
    print("\n==== Hyperparameters")
    hparams.print_values()

    # Load the data and print some statistics.
    vocab_src, vocab_tgt = load_vocabularies(hparams)
    if hparams.share_vocab:
        print("\n==== Vocabulary")
        vocab_src.print_statistics()
    else:
        print("\n==== Source vocabulary")
        vocab_src.print_statistics()
        print("\n==== Target vocabulary")
        vocab_tgt.print_statistics()
    train_data, val_data, opt_data = load_data(hparams, vocab_src=vocab_src, vocab_tgt=vocab_tgt, val_tgt_suffix='.paraphrase' if hparams.paraphrasing_bleu else "")
    print("\n==== Data")
    print(f"Training data: {len(train_data):,} bilingual sentence pairs")
    print(f"Validation data: {len(val_data):,} bilingual sentence pairs")

    # Create the language model and load it onto the GPU if set to do so.
    model, train_fn, validate_fn, _ = create_model(hparams, vocab_src, vocab_tgt)
    optimizers, lr_schedulers = construct_optimizers(
        hparams,
        gen_parameters=model.generative_parameters(),
        inf_z_parameters=model.inference_parameters())
    device = torch.device("cuda:0") if hparams.use_gpu else torch.device("cpu")
    model = model.to(device)

    # Print information about the model.
    param_count_M = model_parameter_count(model) / 1e6
    print("\n==== Model")
    print("Short summary:")
    print(model)
    print("\nAll parameters:")
    for name, param in model.named_parameters():
        print(f"{name} -- {param.size()}")
    print(f"\nNumber of model parameters: {param_count_M:.2f} M")

    # Initialize the model parameters, or load a checkpoint.
    if hparams.model_checkpoint is None:
        print("\nInitializing parameters...")
        initialize_model(model, vocab_tgt[PAD_TOKEN], hparams.cell_type,
                         hparams.emb_init_scale, verbose=True)
    else:
        print(f"\nRestoring model parameters from {hparams.model_checkpoint}...")
        if hparams.forget_decoder:
            #Initialize model first, as there are parts we are not loading
            initialize_model(model, vocab_tgt[PAD_TOKEN], hparams.cell_type,
                             hparams.emb_init_scale, verbose=True)
        loadedpre=torch.load( hparams.model_checkpoint)
        forget_keys=set()
        if hparams.forget_decoder:
            for k in loadedpre:
                if k.split(".")[0] in ['language_model','language_model_tl','lm_init_layer','lm_init_layer_tl','bow_output_layer','bow_output_layer_tl']:
                    forget_keys.add(k)
                if k.startswith("inf_network.normal_layer.scale_layer") or k.startswith("pred_network.normal_layer.scale_layer"):
                    forget_keys.add(k)
        loadeddict={ k:v for k,v in loadedpre.items() if k not in forget_keys }
        print("Loading parameters {}".format(sorted(loadeddict.keys())))
        model.load_state_dict( loadeddict, strict=not hparams.forget_decoder )

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
          vocab_tgt, device, out_dir, train_fn, validate_fn, hparams)

if __name__ == "__main__":
    main()
