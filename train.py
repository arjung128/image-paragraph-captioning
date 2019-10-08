from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

print("Num of GPUs: " + str(torch.cuda.device_count()))

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    # print(opt.alpha)
    print(opt)
    
    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}
    if opt.start_from is not None:
        print("opt.start_from: " + str(opt.start_from))
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    # print("loader.split_ix: " + str(loader.split_ix))
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
        
    # create model
    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)
    # if os.path.isfile("log_sc_vec/model.pth"):
        # model_path = "log_sc_vec/model.pth"
#     if os.path.isfile("alpha=0.1_log_sc_vec/model.pth"):
#         model_path = "alpha=0.1_log_sc_vec/model.pth"
    if os.path.isfile("alpha=0_log_sc/model.pth"):
        model_path = "alpha=0_log_sc/model.pth"
        state_dict = torch.load(model_path)
        dp_model.load_state_dict(state_dict)
        # print("loaded model.")
    dp_model.train()

    # create/load vector model
    vectorModel = models.setup_vectorModel().cuda()
    dp_vectorModel = torch.nn.DataParallel(vectorModel)
    # if os.path.isfile("log_sc_vec/model_vec.pth"):
        # model_vec_path = "log_sc_vec/model_vec.pth"
#     if os.path.isfile("alpha=0.1_log_sc_vec/model.pth"):
#         model_path = "alpha=0.1_log_sc_vec/model.pth"
    if os.path.isfile("alpha=0_log_sc/model_vec.pth"):
        model_vec_path = "alpha=0_log_sc/model_vec.pth"
        state_dict_vec = torch.load(model_vec_path)
        dp_vectorModel.load_state_dict(state_dict_vec)
        # print("loaded model_vec")
    dp_vectorModel.train()

    optimizer = utils.build_optimizer(list(model.parameters()) + list(vectorModel.parameters()), opt)
    update_lr_flag = True

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # Create model
#     model = models.setup(opt).cuda()
#     dp_model = torch.nn.DataParallel(model)
#     model_path = "log_xe_vec/model.pth"
#     state_dict = torch.load(model_path)
#     dp_model.load_state_dict(state_dict)
#     dp_model.train()
    
    # vectorModel
#     vectorModel = models.setup_vectorModel(opt).cuda()
#     dp_vectorModel = torch.nn.DataParallel(vectorModel)
#     dp_vectorModel.train()

    # Loss function
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    # vec_crit = utils.VectorCriterion()
    vec_crit = nn.L1Loss()

    # Optimizer and learning rate adjustment flag
    # optimizer = utils.build_optimizer(model.parameters(), opt)
#     optimizer = utils.build_optimizer(list(model.parameters()) + list(vectorModel.parameters()), opt)
#     update_lr_flag = True

    # Load the optimizer
#     if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
#         optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # create idxs for doc2vec vectors
    with open('paragraphs_image_ids.txt', 'r') as file:
        paragraph_image_ids = file.readlines()

    paragraph_image_ids = [int(i) for i in paragraph_image_ids]
    
    # select corresponding vectors
    with open('paragraphs_vectors.txt', 'r') as the_file:
        vectors = the_file.readlines()

    vectors_list = []
    for string in vectors:
        vectors_list.append([float(s) for s in string.split(' ')])

    vectors_list_np = np.asarray(vectors_list)

    print("Starting training loop!")
        
    # Training loop
    while True:

        # Update learning rate once per epoch
        if update_lr_flag:

            # print("UPDATING")
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        # Load data from train split (0)
        start = time.time()
        data = loader.get_batch('train')
        data_time = time.time() - start
        start = time.time()
        
        # print("data['att_feats'].shape", data['att_feats'].shape)
        # print("data['fc_feats'].shape", data['fc_feats'].shape)
        
        # pad data['att_feats'] axis=1 to have length = 83
        def pad_along_axis(array, target_length, axis=0):

            pad_size = target_length - array.shape[axis]
            axis_nb = len(array.shape)

            if pad_size < 0:
                return a

            npad = [(0, 0) for x in range(axis_nb)]
            npad[axis] = (0, pad_size)

            b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

            return b
        
        data['att_feats'] = pad_along_axis(data['att_feats'], 83, axis=1)

        # Unpack data
        torch.cuda.synchronize()
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        
#         # create model
#         model = models.setup(opt).cuda()
#         dp_model = torch.nn.DataParallel(model)
#         if os.path.isfile("log_sc_vec/model.pth"):
#             model_path = "log_sc_vec/model.pth"
#             state_dict = torch.load(model_path)
#             dp_model.load_state_dict(state_dict)
#             # print("loaded model.")
#         dp_model.train()
        
#         # create/load vector model
#         vectorModel = models.setup_vectorModel().cuda()
#         dp_vectorModel = torch.nn.DataParallel(vectorModel)
#         if os.path.isfile("log_sc_vec/model_vec.pth"):
#             model_vec_path = "log_sc_vec/model_vec.pth"
#             state_dict_vec = torch.load(model_vec_path)
#             dp_vectorModel.load_state_dict(state_dict_vec)
#             # print("loaded model_vec")
#         dp_vectorModel.train()
        
#         optimizer = utils.build_optimizer(list(model.parameters()) + list(vectorModel.parameters()), opt)
#         update_lr_flag = True

#         # Load the optimizer
#         if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
#             optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        
        
#         # create idxs for doc2vec vectors
#         with open('paragraphs_image_ids.txt', 'r') as file:
#             paragraph_image_ids = file.readlines()
            
#         paragraph_image_ids = [int(i) for i in paragraph_image_ids]
        
        idx = []
        for element in data['infos']:
            idx.append(paragraph_image_ids.index(element['id']))
            
#         # select corresponding vectors
#         with open('paragraphs_vectors.txt', 'r') as the_file:
#             vectors = the_file.readlines()
        
#         vectors_list = []
#         for string in vectors:
#             vectors_list.append([float(s) for s in string.split(' ')])
            
#         vectors_list_np = np.asarray(vectors_list)
        
        batch_vectors = vectors_list_np[idx] # MAY NEED TO CONVERT THIS NUMPY ARRAY INTO PYTORCH TENSOR
        
        # Forward pass and loss
        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())
            
        # print("HERE!")
        att_feats_reshaped = att_feats.permute(0, 2, 1).cuda()
        # print(att_feats_reshaped.shape)
        # print(type(att_feats_reshaped))
        # print(type(fc_feats))
        semantic_features = dp_vectorModel(att_feats_reshaped.cuda(), fc_feats) # (10, 2048)
        # print(semantic_features.shape)
        # print(type(semantic_features))
        # print(type(batch_vectors))
        batch_vectors = torch.from_numpy(batch_vectors).float().cuda() # (10, 512)
        # print(type(batch_vectors))
        # print(batch_vectors.shape)
        vec_loss = vec_crit(semantic_features, batch_vectors)
        alpha_ = 0.5
        # loss = loss + (alpha_*vec_loss)

        # Backward pass
        # loss.backward()
        # utils.clip_gradient(optimizer, opt.grad_clip)
        # optimizer.step()
        # train_loss = loss.item()
        # torch.cuda.synchronize()

        # Print 
        total_time = time.time() - start
        if iteration % opt.print_freq == 1:
            print('Read data:', time.time() - start)
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, data_time, total_time))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, np.mean(reward[:,0]), data_time, total_time))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob
            
#         # Evaluate model
#         print("--------------------------------------------------------")
#         print("--------------------EVALUATING MODEL--------------------")
#         print("--------------------------------------------------------")
#         eval_kwargs = {'split': 'val',
#                         'dataset': opt.input_json}
#         eval_kwargs.update(vars(opt))
#         val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)
#         print("--------------------------------------------------------")
#         print("--------------------EVALUATING MODEL--------------------")
#         print("--------------------------------------------------------")

#         # save models
#         checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
#         torch.save(dp_model.state_dict(), checkpoint_path)
#         # print("model saved to {}".format(checkpoint_path))

#         checkpoint_path = os.path.join(opt.checkpoint_path, 'model_vec.pth')
#         torch.save(dp_vectorModel.state_dict(), checkpoint_path)
#         # print("model_vec saved to {}".format(checkpoint_path))

#         optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
#         torch.save(optimizer.state_dict(), optimizer_path)

        # Validate and save model 
        # if (iteration % opt.save_checkpoint_every == 0):
        # if iteration % opt.print_freq == 1:
        if True:

            # Evaluate model
            eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Our metric is CIDEr if available, otherwise validation loss
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            # Save model in checkpoint path 
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(dp_model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))

            # save vec model
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model_vec.pth')
            torch.save(dp_vectorModel.state_dict(), checkpoint_path)
            print("model_vec saved to {}".format(checkpoint_path))

            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            
            
            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            # Save model to unique file if new best model
            if best_flag:
                model_fname = 'model-best-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                infos_fname = 'model-best-i{:05d}-infos.pkl'.format(iteration)
                checkpoint_path = os.path.join(opt.checkpoint_path, model_fname)
                torch.save(dp_model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))

                # best vec
                model_fname_vec = 'model-best-vec-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                checkpoint_path = os.path.join(opt.checkpoint_path, model_fname_vec)
                torch.save(dp_vectorModel.state_dict(), checkpoint_path)
                print("model_vec saved to {}".format(checkpoint_path))

                with open(os.path.join(opt.checkpoint_path, infos_fname), 'wb') as f:
                    cPickle.dump(infos, f)
                
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
print(type(opt))
print(opt)
train(opt)
