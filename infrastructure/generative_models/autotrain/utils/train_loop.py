import time
import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(import_path)
# sys.path.append('~')
import torch.nn as nn
from generative_models.Process import *
from generative_models.Batch import create_masks
from generative_models.inference import validate_docking
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from generative_models.Optim import CosineWithRestarts
def KLAnnealer(opt, epoch):
    beta = opt.KLA_ini_beta + opt.KLA_inc_beta * ((epoch + 1) - opt.KLA_beg_epoch)
    return beta

def loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var):
    RCE_mol = F.cross_entropy(preds_mol.contiguous().view(-1, preds_mol.size(-1)), ys_mol, ignore_index=opt.trg_pad, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if opt.use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='sum')
        loss = RCE_mol + RCE_prop + beta * KLD
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_mol + beta * KLD
    return loss, RCE_mol, RCE_prop, KLD



def train_model(model : nn.Module, opt,case='Alzmhr'):
    """Training loop function.

    Args:
        model (nn.Module): PyTroch model transformer for training
        opt (NameSpace): NameSpace with training parameters
    """
    print("training model...")
    model.train()

    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
    history_cond_score, history_valid, history_novelty, history_duplicates, history_diversity, history_intersted_cond = [], [], [], [], [], []
    history_epoch, history_beta, history_lr = [], [], []
    history_total_loss, history_RCE_mol_loss, history_RCE_prop_loss, history_KLD_loss = [], [], [], []
    history_total_loss_te, history_RCE_mol_loss_te, history_RCE_prop_loss_te, history_KLD_loss_te = [], [], [], []
    QED,Synthetic_Accessibility,PAINS,SureChEMBL,Glaxo,Brenk,IC50 = [],[],[],[],[],[],[],
    history_props = {'QED':[],'Synthetic Accessibility':[],'PAINS':[],'SureChEMBL':[],'Glaxo':[],'Brenk':[],'IC50':[]}

    beta = 0
    current_step = 0
    for epoch in range(opt.epochs):
        total_loss, RCE_mol_loss, RCE_prop_loss, KLD_loss= 0, 0, 0, 0
        accum_train_printevery_n, accum_test_n, accum_test_printevery_n = 0, 0, 0

        if opt.floyd is False:
            print("     {TR}   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), f'{opt.save_folder_name}/weights/model_weights')

        # KL annealing
        if opt.use_KLA == True:
            if epoch + 1 >= opt.KLA_beg_epoch and beta < opt.KLA_max_beta:
                beta = KLAnnealer(opt, epoch)
        else:
            beta = 1

        for i, batch in enumerate(opt.train):
            current_step += 1
            src = batch.src.transpose(0, 1).to(opt.device)
            trg = batch.trg.transpose(0, 1).to(opt.device)
            trg_input = trg[:, :-1]

            cond = torch.stack([batch.__dict__[i] for i in opt.conditions]).transpose(0, 1).to(opt.device)
            src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
            preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
            ys_mol = trg[:, 1:].contiguous().view(-1)
            ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

            opt.optimizer.zero_grad()
            loss, RCE_mol, RCE_prop, KLD = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)
            loss.backward()
            opt.optimizer.step()
            if opt.lr_scheduler == "SGDR":
                opt.sched.step()

            if opt.lr_scheduler == "WarmUpDefault":
                head = np.float64(np.power(np.float64(current_step), -0.5))
                tail = np.float64(current_step) * np.power(np.float64(opt.lr_WarmUpSteps), -1.5)
                lr = np.float64(np.power(np.float64(opt.d_model), -0.5)) * min(head, tail)
                for param_group in opt.optimizer.param_groups:
                    param_group['lr'] = lr

            for param_group in opt.optimizer.param_groups:
                current_lr = param_group['lr']

            total_loss += loss.item()
            RCE_mol_loss += RCE_mol.item()
            RCE_prop_loss += RCE_prop.item()
            KLD_loss += KLD.item()

            accum_train_printevery_n += len(batch)
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss /accum_train_printevery_n
                 avg_RCE_mol_loss = RCE_mol_loss /accum_train_printevery_n
                 avg_RCE_prop_loss = RCE_prop_loss /accum_train_printevery_n
                 avg_KLD_loss = KLD_loss /accum_train_printevery_n
                 if opt.floyd is False:
                    print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr), end='\r')
                 else:
                    print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr))
                 accum_train_printevery_n, total_loss, RCE_mol_loss, RCE_prop_loss, KLD_loss = 0, 0, 0, 0, 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), f'{opt.save_folder_name}/weights/model_weights')
                cptime = time.time()

        print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_mol = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, avg_RCE_mol_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr))
        #try:
        cond_score, valid, novelty, duplicates, of_ineterested_cond, diversity,d_pop = validate_docking(
            opt=opt,
            model=model,
            SRC=opt.SRC,
            TRG=opt.TRG,
            n_samples=opt.n_samples,
            spec_conds=[-10,0.99,1,0,0,0,0,1],
            case=case)
        
        [history_props[i].append(d_pop[i]) for i in d_pop]
        history_epoch.append(epoch + 1)
        history_beta.append(beta)
        history_total_loss.append(avg_loss)
        history_RCE_mol_loss.append(avg_RCE_mol_loss)
        history_RCE_prop_loss.append(avg_RCE_prop_loss)
        history_KLD_loss.append(avg_KLD_loss)
        history_cond_score.append(cond_score)
        history_valid.append(valid)
        history_lr.append(current_lr)
        history_novelty.append(novelty)
        history_duplicates.append(duplicates)
        history_diversity.append(diversity)
        history_intersted_cond.append(of_ineterested_cond)
        data_dict = dict({"epochs": history_epoch, "beta": history_beta, "lr": history_lr, "total_loss": history_total_loss,
                "RCE_mol_loss": history_RCE_mol_loss,
                "RCE_prop_loss": history_RCE_prop_loss, "KLD_loss": history_KLD_loss,
                'cond_score': history_cond_score,
                'valid': history_valid,
                'novelty': history_novelty,
                'duplicates': history_duplicates, 'diversity': history_diversity,
                'SA_of_ineterested_cond': history_intersted_cond},**history_props)
        history = pd.DataFrame(data_dict
            )
        history.to_csv(f'{opt.save_folder_name}/weights/History_{opt.latent_dim}_epo={opt.epochs}_{time.strftime("%Y%m%d")}.csv',index=True)

        # Export weights every epoch
        if not os.path.isdir('{}'.format(opt.save_folder_name)):
            os.mkdir('{}'.format(opt.save_folder_name))
        if not os.path.isdir('{}/epo{}'.format(f'{opt.save_folder_name}/weights', epoch + 1)):
            os.mkdir('{}/epo{}'.format(f'{opt.save_folder_name}/weights', epoch + 1))
        torch.save(model.state_dict(), f'{opt.save_folder_name}/weights/epo{epoch+1}/model_weights')
