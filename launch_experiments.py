import subprocess
from itertools import product


OUTDIR = "/home/ubuntu/logs_russian_roulette_adversarial_training/"
EPOCHS = 90
LOG_ITERS = 3

params = {
    'eps': [0.25, 0.5, 1., 2.],
    'stop-prob': [1./3, 1./7, 1./20, 0],
    'arch': ['resnet18', 'resnet50'],
    'attack-steps': [3, 7, 10, 20]
}

for arch, eps, attack_steps, stop_prob in product(params['arch'], 
                                                  params['eps'],
                                                  params['attack-steps'],
                                                  params['stop-prob'], 
                                                  ): 
    attack_lr = 2.5 * eps / attack_steps
    command = f"""python -m robustness.main --dataset cifar \
        --out-dir {OUTDIR} \
        --adv-train 1 \
        --stop-prob {stop_prob} \
        --attack-steps {attack_steps}
        --constraint 2 \
        --eps {eps} \
        --attack-lr {attack_lr} \
        --arch {arch} \
        --epoch {EPOCHS} \
        --log-iters {LOG_ITERS}
        """
    process = subprocess.run(command.split())