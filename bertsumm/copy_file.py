import shutil

shutil.rmtree("../output/summary/bert")
shutil.copytree("data/result/bert", "../output/summary/bert")