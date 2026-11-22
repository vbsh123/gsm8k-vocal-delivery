"""Summarize only the new Phi L11 answer strengths -1.5 and -2, on all six tasks."""
from phi4_l11_summary import main


if __name__ == '__main__':
    main(default_strengths=[-1.5, -2],
         default_datasets=['gsm8k', 'multiarith', 'svamp', 'bbh3', 'bbh5', 'ifeval'])
