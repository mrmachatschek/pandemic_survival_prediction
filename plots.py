import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def bar_charts_categorical(df, feature, target):
    palette = ['lightcoral', 'deepskyblue']
    cont_tab = pd.crosstab(df[feature], df[target], margins = True)
    categories = cont_tab.index[:-1]

    fig = plt.figure(figsize=(11, 5))

    plt.subplot(121)
    p1 = plt.bar(categories, cont_tab.iloc[:-1, 0].values, 0.55, color=palette[1])
    p2 = plt.bar(categories, cont_tab.iloc[:-1, 1].values, 0.55, bottom=cont_tab.iloc[:-1, 0], color=palette[0])
    plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
    plt.title("Frequency bar chart")
    plt.xlabel(feature)
    plt.ylabel("$Frequency$")

    # auxiliary data for 122
    obs_pct = np.array([np.divide(cont_tab.iloc[:-1, 0].values, cont_tab.iloc[:-1, 2].values),
                        np.divide(cont_tab.iloc[:-1, 1].values, cont_tab.iloc[:-1, 2].values)])

    plt.subplot(122)
    p1 = plt.bar(categories, obs_pct[0], 0.55, color=palette[1])
    p2 = plt.bar(categories, obs_pct[1], 0.55, bottom=obs_pct[0], color=palette[0])
    plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
    plt.title("Proportion bar chart")
    plt.xlabel(feature)
    plt.ylabel("$p$")

    plt.show()

def box_plots(df):
    palette = ['lightcoral', 'deepskyblue']

    f = plt.figure(figsize=(11, 8))
    gs = f.add_gridspec(2,1)

    ax = f.add_subplot(gs[0, 0])
    sns.boxplot(data=df, x = 'Birthday_year', y = 'Deceased', orient = 'h', palette = palette)

    ax = f.add_subplot(gs[1, 0])
    sns.boxplot(data=df, x = 'Medical_Expenses_Family', y = 'Deceased', orient = 'h', palette = palette)


def pearson_correlation_plot(df):
    pearson_corr_matrix=df.corr('pearson').round(decimals=2)
    sns.set(rc={'figure.figsize':(7,4)})
    sns.heatmap(pearson_corr_matrix,
        xticklabels=pearson_corr_matrix.columns,
        yticklabels=pearson_corr_matrix.columns,
        annot=True,
        linewidths=.5,
        vmin = -1,
        vmax = 1,
        cmap=sns.diverging_palette(-10, 240, sep=70, n=7))


def family_sizes(df):
    families = df.groupby(["Family_Case_ID"])["Family_Case_ID"].count().reset_index(name='count').sort_values(by='count')
    sns.set(rc={'figure.figsize':(6,4)})
    ax = families.plot(x='Family_Case_ID', y ='count' , kind="hist",legend= False, title = "Families by number of members",bins=7, color = 'gray')
    ax.set_xlabel("Number of family members infected")
    ax.set_ylabel("Frequency")


def plot_numerical_variables(df):
    numerical_var = ["Severity",
                 "Parents or siblings infected",
                 "Wife/Husband or children infected",
                 "Medical_Expenses_Family"]

    sns.set(rc={'figure.figsize':(10,8)})
    df[numerical_var].hist(bins=15, layout=(2, 2), xlabelsize=8, ylabelsize=8, color = 'gray');
