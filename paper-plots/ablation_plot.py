import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize=(30, 5), dpi=500)

# set height of bar
y1 = [0.8432, 0.8706, 0.8351, 0.3735, 0.3063, 0.4572, 0.8801,
      0.7741, 0.8962, 0.7978, 0.6220, 0.3705, 0.3066, 0.7463, 0.5987]
y2 = [0.5405, 0.4353, 0.4432, 0.2524, 0.5754, 0.6978, 0.8431,
      0.6870, 0.8205, 0.8203, 0.8203, 0.5445, 0.6153, 0.8718, 0.6559]
y3 = [0.8405, 0.8608, 0.8297, 0.3670, 0.3083, 0.6987, 0.7831,
      0.7221, 0.8760, 0.8530, 0.8289, 0.5462, 0.6165, 0.8889, 0.6589]
y4 = [0.8432, 0.8706, 0.8351, 0.3735, 0.5754, 0.6978, 0.8831,
      0.7741, 0.8962, 0.8557, 0.8300, 0.5468, 0.6209, 0.9066, 0.6619]

# Set position of bar on X axis
br1 = np.arange(len(y1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]


# Make the plot
plt.bar(br1, y1, color='#e41a1c', width=barWidth,
        edgecolor='black', label='GloGNN-na', hatch='xx', lw=1)
# plt.bar(br1, y1, color ='none', width = barWidth,
#   edgecolor ='black', zorder=1, lw = 2)

plt.bar(br2, y2, color='#377eb8', width=barWidth,
        edgecolor='black', label='GloGNN-nf', hatch='//', lw=1)
plt.bar(br3, y3, color='#4daf4a', width=barWidth,
        edgecolor='black', label='GloGNN-nl', hatch='..', lw=1)
plt.bar(br4, y4, color='#984ea3', width=barWidth,
        edgecolor='black', label='GloGNN', hatch='--', lw=1)

plt.xlim((-0.2, 15))
plt.ylim((0, 1))

# Adding Xticks
plt.xlabel('Datasets', fontweight='bold', fontsize=15)
plt.ylabel('Accuracy/AUC score', fontweight='bold', fontsize=15)
plt.xticks([r + 2 * barWidth for r in range(len(y1))],
           ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Squirrel', 'Chameleon', 'Cora', 'Citeseer', 'Pubmed',
            'Penn94', 'pokec', 'arXiv', 'snap', 'genius', 'twitch'])

plt.legend(loc=0, fontsize=8)
plt.savefig('ablation.eps')
# plt.show()
