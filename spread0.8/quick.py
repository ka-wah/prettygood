import pandas as pd

df = pd.read_parquet('spread0.8/dhinput.parquet')

# show histogram distribution of mid_t
df['mid_t'].hist(bins=500)
import matplotlib.pyplot as plt
plt.title('Histogram of mid_t')
plt.xlabel('mid_t')
plt.ylabel('Frequency')
# max x axis of 5000
plt.xlim(0, 5000)
plt.show()