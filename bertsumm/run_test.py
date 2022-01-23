import main
import numpy as np
seed = np.random.randint(0, 1000)
print(seed)
# use_bert(seed, use_test=True)
main.use_mlm(seed, use_test=True)