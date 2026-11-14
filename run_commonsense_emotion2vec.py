"""Classify CommonsenseQA audio and report saved DIFFA/Qwen/Kimi accuracy on agreed pairs."""

import sys
from run_arc_emotion2vec import main


if __name__=='__main__':main(['--task','commonsenseqa',*sys.argv[1:]])
