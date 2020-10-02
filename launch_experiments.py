import subprocess
from itertools import product


OUTDIR = "/home/ubuntu/logs_russian_roulette_adversarial_training/loss_russian_roulette/"
EPOCHS = 90
LOG_ITERS = 3

params = {
    'eps': [0.25, 0.5, 1., 2.],
    'stop-prob': [1./3, 1./20],
    'arch': ['resnet18', 'resnet50'],
}

for arch, eps, stop_prob in product(params['arch'], 
                                                  params['eps'],
                                                  params['stop-prob'], 
                                                  ): 
    attack_lr = 2.5 * eps * stop_prob
    command = f"""python -m robustness.main --dataset cifar \
        --out-dir {OUTDIR} \
        --adv-train 1 \
        --attack-steps 7 \
        --constraint 2 \
        --eps {eps} \
        --attack-lr {attack_lr} \
        --arch {arch} \
        --epoch {EPOCHS} \
        --log-iters {LOG_ITERS} \
        --russian-roulette \
        --stop-prob {stop_prob}
        """
    process = subprocess.run(command.split())