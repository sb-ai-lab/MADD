import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn

from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from scripts.layers import Generator, RecurrentDiscriminator
from scripts.tokenizer import Tokenizer
from torch.utils.tensorboard import SummaryWriter


writer=SummaryWriter()

class MolGen(nn.Module):

    def __init__(self, data, hidden_dim=128, lr=1e-3, device='cpu'):
        """[summary]

        Args:
            data (list[str]): [description]
            hidden_dim (int, optional): [description]. Defaults to 128.
            lr ([type], optional): learning rate. Defaults to 1e-3.
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.
        """
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(data)

        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size - 1,
            start_token=self.tokenizer.start_token - 1,  # no need token
            end_token=self.tokenizer.end_token - 1,
        ).to(device)

        self.discriminator = RecurrentDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)

        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(), lr=lr)

        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)

        self.b = 0.  # baseline reward
        self.total_step = 0

    def sample_latent(self, batch_size):
        """Sample from latent space

        Args:
            batch_size (int): number of samples

        Returns:
            torch.Tensor: [batch_size, self.hidden_dim]
        """
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y):
        """Discriminator loss

        Args:
            x (torch.LongTensor): input sequence [batch_size, max_len]
            y (torch.LongTensor): sequence label (zeros from generatoe, ones from real data)
                                  [batch_size, max_len]

        Returns:
            loss value
        """

        y_pred, mask = self.discriminator(x).values()

        loss = F.binary_cross_entropy(
            y_pred, y, reduction='none') * mask

        loss = loss.sum() / mask.sum()

        return loss

    def train_step(self, x):
        """One training step

        Args:
            x (torch.LongTensor): sample form real distribution
        """

        batch_size, len_real = x.size()

        # create real and fake labels
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=100)
        x_gen, log_probs, entropies = generator_outputs.values()

        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        #####################
        # Train Discriminator
        #####################

        self.discriminator_optim.zero_grad()

        # disc fake loss
        fake_loss = self.discriminator_loss(x_gen, y_gen)

        # disc real loss
        real_loss = self.discriminator_loss(x_real, y_real)

        # combined loss
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()
        writer.add_scalar('discr_loss_v4', discr_loss.cpu().detach().numpy(), self.total_step)
        print('discr_loss', discr_loss.cpu().detach().numpy())
        # clip grad
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # update params
        self.discriminator_optim.step()

        # ###############
        # Train Generator
        # ###############

        self.generator_optim.zero_grad()

        # prediction for generated x
        y_pred, y_pred_mask = self.discriminator(x_gen).values()

        # Reward (see the ref paper)
        R = (2 * y_pred - 1)

        # reward len for each sequence
        lengths = y_pred_mask.sum(1).long()

        # list of rew of each sequences
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # compute - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):

            # substract the baseline
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        # mean loss + entropy reg
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * 0.01 / batch_size
            
        writer.add_scalar('gen_loss_v4', generator_loss.cpu().detach().numpy(), self.total_step)
        print('generator_los', generator_loss.cpu().detach().numpy())
        # baseline moving average
        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):
        """create a dataloader

        Args:
            data (list[str]): list of molecule smiles
            batch_size (int, optional): Defaults to 128.
            shuffle (bool, optional): Defaults to True.
            num_workers (int, optional): Defaults to 5.

        Returns:
            torch.data.DataLoader: a torch dataloader
        """

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    # def train_n_steps(self, train_loader, max_step=10000, evaluate_every=50):
    #     """Train for max_step steps
    #
    #     Args:
    #         train_loader (torch.data.DataLoader): dataloader
    #         max_step (int, optional): Defaults to 10000.
    #         evaluate_every (int, optional): Defaults to 50.
    #     """
    #
    #     iter_loader = iter(train_loader)
    #
    #     # best_score = 0.0
    #     for step in range(max_step):
    #         print('Step is',step)
    #
    #         try:
    #             batch = next(iter_loader)
    #         except:
    #             iter_loader = iter(train_loader)
    #             batch = next(iter_loader)
    #
    #         # model update
    #         self.train_step(batch)
    #
    #         if step % evaluate_every == 0:
    #
    #             self.eval()
    #             score = self.evaluate_n(100)
    #             self.train()
    #
    #             # if score > best_score:
    #             #     self.save_best()
    #             #     print('saving')
    #             #     best_score = score
    #
    #             print(f'valid = {score: .2f}')
    
    def train_n_steps(self, train_loader, max_step=10000, evaluate_every=50):
        """Train for max_step steps

        Args:
            train_loader (torch.data.DataLoader): dataloader
            max_step (int, optional): Defaults to 10000.
            evaluate_every (int, optional): Defaults to 50.
        """

        #iter_loader = iter(train_loader)

        best_score = 0.0
        for epoch in range(max_step):
            print('#'*12,f'Epoch:{epoch}','#'*12,sep='\n')
            for i, batch in enumerate(train_loader):
                print('Batch #',i)
        # for step in range(max_step):
        #     print('Step is',step)
        #
        #     try:
        #         batch = next(iter_loader)
        #     except:
        #         iter_loader = iter(train_loader)
        #         batch = next(iter_loader)

            # model update
                self.train_step(batch)
                self.total_step += 1

                if i % evaluate_every == 0:

                    self.eval()
                    score = self.evaluate_n(100)
                    self.train()

                    writer.add_scalar('valid_v4', score, self.total_step)
                    if score > best_score:
                        torch.save(self.generator.state_dict(), 'v4_gan_best_generator.pt')
                        torch.save(self.discriminator.state_dict(), 'v4_gan_best_discriminator.pt')
                        print('saving')
                        best_score = score

                    print(f'valid = {score: .2f}')

    def get_mapped(self, seq):
        """Transform a sequence of ids to string

        Args:
            seq (list[int]): sequence of ids

        Returns:
            str: string output
        """
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def generate_n(self, n):
        """Generate n molecules

        Args:
            n (int)

        Returns:
            list[str]: generated molecules
        """

        z = torch.randn((n, self.hidden_dim)).to(self.device)

        x = self.generator(z)['x'].cpu()
        #pd.DataFrame
        lenghts = (x > 0).sum(1)

        # l - 1 because we exclude end tokens
        return [self.get_mapped(x[:l-1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n, path = None):
        """Evaluation: frequence of valid molecules using rdkit

        Args:
            n (int): number of sample

        Returns:
            float: requence of valid molecules
        """

        pack = self.generate_n(n)

        print(pack[:2])

        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])
        if path is not None:
            pd.DataFrame(data={'0': pack, 'val_check': list(valid)}).to_csv(path)
        return valid.mean()