import torch
from .base_model import BaseModel
from . import networks
from . import losses
import torch.nn.functional as F
from .iqa import *

class Pix2PixModel(BaseModel):

    def __init__(self, opt):
   
        BaseModel.__init__(self, opt)
        # losses to save/display
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']  
        if self.isTrain:
            if opt.lambda_L1 != 0:
                self.loss_names.append('G_L1')
            if opt.lambda_L2 != 0:
                self.loss_names.append('G_L2')
            if opt.lambda_desc != 0:
                self.loss_names.append('G_desc')
            if opt.lambda_ms_ssim != 0:
                self.loss_names.append('G_ms_ssim')
            if opt.lambda_ssim != 0:
                self.loss_names.append('G_ssim')
            if opt.lambda_edge != 0:
                self.loss_names.append('G_edge')
            if opt.lambda_feat != 0:
                self.loss_names.append('G_feat')
            if opt.lambda_dists != 0:
                self.loss_names.append('G_dists')
            if opt.lambda_iqa != 0:
                self.loss_names.append('G_iqa')
            
        # images to save/display. 
        self.visual_names = ['real_A', 'fake_B', 'real_B']
     
        # models to save 
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(input_nc = opt.input_nc, 
                                      output_nc = opt.output_nc, 
                                      ngf = opt.ngf, 
                                      netG = opt.netG, 
                                      norm = opt.norm,
                                      use_dropout= not opt.no_dropout, 
                                      init_type = opt.init_type, 
                                      init_gain=opt.init_gain, 
                                      gpu_ids= self.gpu_ids,
                                      att_type = opt.att_type,
                                      kan_norm = opt.kan_norm,
                                      
                                      )

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(input_nc = opt.input_nc + opt.output_nc, 
                                          ndf = opt.ndf, 
                                          netD = opt.netD,
                                          n_layers_D = opt.n_layers_D, 
                                          norm = opt.norm, 
                                          init_type= opt.init_type, 
                                          init_gain = opt.init_gain, 
                                          gpu_ids= self.gpu_ids, 
                                          dropout = opt.use_D_dropout,
                                          getIntermFeat=True,
                                          num_D=opt.num_D,

                                          )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            if opt.lambda_L1 != 0:
                self.criterionL1 = torch.nn.L1Loss()
            if opt.lambda_L2 != 0:
                self.criterionL2 = torch.nn.MSELoss()
            
            if opt.lambda_ms_ssim != 0:
                self.criterion_ms_ssim = losses.MS_SSIM_loss()
            if opt.lambda_ssim != 0:
                self.criterion_ssim = losses.SSIM_loss()
            if opt.lambda_edge != 0:
                self.criterion_edge = losses.edge_based_loss(gpu_ids=self.opt.gpu_ids,
                    detector=self.opt.edge_detector,
                    kernel_size=self.opt.kernel_size,
                    loss=self.opt.edge_losses)
              
            if opt.lambda_feat != 0:
                self.criterion_feat = losses.FeatureLossMSD()
            if opt.lambda_dists != 0:
                self.criterion_dists = losses.DISTS_Loss(gpu_ids=self.opt.gpu_ids)
            if opt.lambda_iqa!=0:
                self.criterion_iqa = IQA(gpu_ids=self.opt.gpu_ids, iqa_metric = opt.iqa_name)
            
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
         
        
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
       
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        output = self.netG(self.real_A)
        self.fake_B = output


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # Fake; stop backprop to the generator by detaching fake_B
        # if len(self.fake_B)==1:
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self,opt,current_epoch=0):
        # First, G(A) should fake the discriminator
        # if len(self.fake_B)==1:
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # else:
        #     fake_AB = torch.cat((self.real_A, self.fake_B['y_rgb']),1)
        pred_fake = self.netD(fake_AB)
  
        # GAN loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G = self.loss_G_GAN
        
        # L1
        if opt.lambda_L1 != 0:
            if opt.changeL1after != 0:
                lambda_L1 = opt.lambda_L1_after_n_epochs if current_epoch > opt.changeL1after else opt.lambda_L1
            else:
                lambda_L1 = opt.lambda_L1
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1
            self.loss_G = self.loss_G + self.loss_G_L1

        # L2
        if opt.lambda_L2 != 0:
           self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_L2
           self.loss_G = self.loss_G + self.loss_G_L2

        # MS-SSIM Loss
        
        if opt.lambda_ms_ssim != 0:
            if current_epoch < opt.usems_ssimbefore:
                self.loss_G_ms_ssim = self.criterion_ms_ssim(self.fake_B, self.real_B) * self.opt.lambda_ms_ssim
            else:
                self.loss_G_ms_ssim = 0

            self.loss_G = self.loss_G + self.loss_G_ms_ssim

        # SSIM Loss
        if opt.lambda_ssim != 0:
            self.loss_G_ssim = self.criterion_ssim(self.fake_B, self.real_B) * self.opt.lambda_ssim 
            self.loss_G = self.loss_G + self.loss_G_ssim    

        # Edge Loss
        if opt.lambda_edge != 0:
            if current_epoch > opt.useedgeafter:
                if opt.compare_edge_with == 'rgb':
                    self.loss_G_edge = self.criterion_edge(generated_rgb = self.fake_B, nir_image = None, real_rgb = self.real_B) * self.opt.lambda_edge 
                else:
                    self.loss_G_edge = self.criterion_edge(generated_rgb = self.fake_B, nir_image = self.real_A, real_rgb = None) * self.opt.lambda_edge 
            
            else:
                self.loss_G_edge = 0
           
            self.loss_G = self.loss_G + self.loss_G_edge  

        # Feature Matching Loss
        if opt.lambda_feat != 0:
            pred_real = self.netD(torch.cat((self.real_A, self.real_B), 1))
            pred_fake = self.netD(torch.cat((self.real_A, self.fake_B), 1))
            self.loss_G_feat = self.criterion_feat(pred_real, pred_fake, opt.n_layers_D, opt.num_D) * self.opt.lambda_feat

            self.loss_G = self.loss_G + self.loss_G_feat

        # DISTS Loss
        if opt.lambda_dists != 0:
            if current_epoch < opt.usedistsbefore:
                self.loss_G_dists = self.criterion_dists(generated_image = self.fake_B, target_image = self.real_B) * opt.lambda_dists
            else:
                self.loss_G_dists = 0
            self.loss_G = self.loss_G + self.loss_G_dists
            
        # IQA Loss
        if opt.lambda_iqa != 0:
            if current_epoch>opt.useiqaafter:
                self.loss_G_iqa = self.criterion_iqa(generated_image = self.fake_B, target_image = self.real_B, NR = opt.trueNR, L2= opt.useL2IQA) * opt.lambda_iqa
            else:
                self.loss_G_iqa = 0
            
            self.loss_G = self.loss_G + self.loss_G_iqa
            
        self.loss_G.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)


    def optimize_parameters(self, current_epoch=0):
        self.forward()
        if current_epoch % self.opt.train_D_every == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)

            self.optimizer_D.step()
        else:
            self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        # self.backward_G(opt=self.opt)
        self.backward_G(opt=self.opt, current_epoch=current_epoch)

        self.optimizer_G.step()

   