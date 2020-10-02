import torch as ch
from scipy.stats import geom

from .attacker import AttackerModel
from .tools import helpers


class RussianRouletteTrainer(ch.nn.Module):
    """
    Wrapper class for adversarially training a model using Russian Roulette.
    """

    def __init__(self, model, dataset):
        super(RussianRouletteTrainer, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.dataset = dataset
        self.attacker = AttackerModel(model, dataset)
        self.criterion = ch.nn.CrossEntropyLoss()
    
    def forward(self, inp, target=None, make_adv=False, 
                with_latent=False, stop_probability=1./20,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):
        """
        Outputs an unbiased stochastic estimator of the min max loss,
        using an underlying sequence of adversarial examples.
        This sequence comes from an AttackerModel.
        """
        if ("russian_roulette" not in attacker_kwargs) or (not attacker_kwargs["russian_roulette"]):  
            return self.attacker(inp, target, make_adv=make_adv, 
                                 with_latent=with_latent, fake_relu=fake_relu,
                                 no_relu=no_relu, **attacker_kwargs)        

        iterations = geom.rvs(stop_probability, loc=-1)

        initial_output = self.model(inp, with_latent=with_latent, fake_relu=fake_relu,
                                  no_relu=no_relu)
        loss = self.criterion(initial_output, target)
        if iterations:
            attacker_kwargs['iterations'] = iterations - 1

            self.eval()
            _, prev_adv = self.attacker(inp, target, make_adv=True, **attacker_kwargs)
            attacker_kwargs['iterations'] = 1
            normalized_prev_adv = self.normalizer(prev_adv)
            _, adv = self.attacker(prev_adv, target, make_adv=True, **attacker_kwargs)
            normalized_adv = self.normalizer(adv)

            # Get the losses with gradients
            self.train() 
            prev_output = self.model(normalized_prev_adv, with_latent=with_latent, fake_relu=fake_relu,
                                  no_relu=no_relu)
            output = self.model(normalized_adv, with_latent=with_latent, fake_relu=fake_relu,
                                no_relu=no_relu)
            
            loss += (1-stop_probability) ** (-iterations) / stop_probability *\
                     (self.criterion(output, target) - self.criterion(prev_output, target))

        return loss
            
