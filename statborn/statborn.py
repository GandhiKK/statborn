import numpy as _np
import pandas as _pd
import re
import matplotlib.pyplot as _plt
import matplotlib as _mpl
import seaborn as _sns
import scipy as _scipy
import warnings as _w
import math as _math 
from bioinfokit.analys import stat as _stat
from scipy.stats import chi2_contingency as _chi2cont
from statsmodels.graphics.mosaicplot import mosaic as _msc
from statsmodels.stats.contingency_tables import mcnemar
import scikit_posthocs as _sp
import textwrap as _tp
_w.filterwarnings("ignore")




# normality check
class Normality:
    """
    Class for inspect data for normality

    ...

    Init Attributes
    ----------
    df : DataFrame
        data frame
    col : str
        column label

    Methods
    -------
    normality_tests():
        Tests the null hypothesis that a sample comes from a normal distribution. It has several tests:
            1. Shapiro Wilk test
            2. Kolmogorov Smirnov test
            3. Pearson's test
    param_tests():
        This function ses skew and kurtosis tests
    diag_tests(palette='pastel', save=False, path_no_title=None):
        Data visualization for detecting non-normal samples:
            1. Histogram
            2. Box Plot
            3. QQ Plot
    combine_tests()
        This function combines all of the tests
    """
    
    _nround = 6
    
    def __init__(self, df=None, col=None):
        self._df = _pd.DataFrame(df[col])
        self._original = df[col]
        self._shape = self._df.shape
        self._nrows = self._df.shape[0]
        self._title = self._df.columns.values[0]

    # Shapiro Wilk test  
    def __shapiro(self, stat):
        try:
            stat_SW, p_SW = _scipy.stats.shapiro(self._df)
        except ValueError:
            stat_SW, p_SW = _np.NaN, _np.NaN
        except IndexError: #todo
            stat_SW, p_SW = _np.NaN, _np.NaN
        stat['Shapiro–Wilk'] = [round(stat_SW, self._nround), round(p_SW, self._nround), 'True' if p_SW > 0.05 else 'False']
        return stat
    
    # Kolmogorov Smirnov test
    def __ks(self, stat):
        try:
            stat_KS, p_KS = _scipy.stats.kstest(self._df, 'norm', args=(_np.mean(self._df), _np.std(self._df)))
        except ValueError:
            stat_KS, p_KS = _np.NaN, _np.NaN
        except IndexError: #todo
            stat_KS, p_KS = _np.NaN, _np.NaN
        stat['Kolmogorov-Smirnov'] = [round(stat_KS, self._nround), round(p_KS, self._nround), 'True' if p_KS > 0.2 else 'False']
        return stat
    
    # Pearson's test
    def __pearson(self, stat):
        try:
            stat_P, p_P = _scipy.stats.normaltest(self._original)
        except ValueError:
            stat_P, p_P = _np.NaN, _np.NaN
        except IndexError: #todo
            stat_P, p_P = _np.NaN, _np.NaN
        stat['Pearson'] = [round(stat_P, self._nround), round(p_P, self._nround), 'True' if p_P > 0.05 else 'False']  
        return stat
    
    # skew test
    def __skew(self, stat):
        try: 
            stat_skew, p_skew = _scipy.stats.skewtest(self._original)
        except ValueError:
            stat_skew, p_skew = _np.NaN, _np.NaN
        except IndexError: #todo
            stat_skew, p_skew = _np.NaN, _np.NaN
        stat['Skew test'] = [round(stat_skew, self._nround), round(p_skew, self._nround), 'True' if p_skew > 0.05 else 'False']
        return stat
    
    # kurtosis test
    def __kurtosis(self, stat):
        try:
            stat_kurt, p_kurt = _scipy.stats.kurtosistest(self._original)
        except ValueError:
            stat_kurt, p_kurt = _np.NaN, _np.NaN
        except IndexError: #todo
            stat_kurt, p_kurt = _np.NaN, _np.NaN
        stat['Kurtosis test'] = [round(stat_kurt, self._nround), round(p_kurt, self._nround), 'True' if p_kurt > 0.05 else 'False']
        return stat
    
    # SW, KS, P tests
    def normality_tests(self):
        """
        This function tests the null hypothesis that a sample comes from a normal distribution. It has several tests:
            1. Shapiro Wilk test
            2. Kolmogorov Smirnov test
            3. Pearson's test

        Parameters
        ----------
        None
        
        Returns
        -------
        DataFrame
        """
        
        statistics = _pd.DataFrame(index=['Test statistics', 'p value', 'Check'])
        if self._nrows > 50:
            statistics = self.__ks(statistics)
        statistics = self.__shapiro(statistics)
        statistics = self.__pearson(statistics)      
        return statistics
    
    # skew/kurtosis
    def param_tests(self):
        """
        This function uses skew and kurtosis tests

        Parameters
        ----------
        None
        
        Returns
        -------
        DataFrame
        """
        statistics = _pd.DataFrame(index=['Test statistics', 'p value', 'Check'])
        statistics = self.__skew(statistics)
        statistics = self.__kurtosis(statistics)
        return statistics
    
    # plots
    def diag_tests(self, palette='pastel', save=False, path_no_title=None):
        """
        This function uses data visualization for detecting non-normal samples:
            1. Histogram
            2. Box Plot
            3. QQ Plot

        Parameters
        ----------
        palette : str, default 'pastel'
            palette for visualizations
        save : bool, default False
            plots saving
        path_no_title : str, default ``None``
            path to the save folder
        
        Returns
        -------
        None
        """
        
        _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)        
        ax = _sns.histplot(data=self._df, x=self._title, kde=False, stat='density', bins=round(1+3.31*_np.log10(self._nrows)))
        x0, x1 = ax.get_xlim()
        x_pdf = _np.linspace(x0, x1, 100)
        y_pdf = _scipy.stats.norm.pdf(x_pdf, loc=self._df.mean(), scale=self._df.std())
        ax.plot(x_pdf, y_pdf, 'r', lw=2)       
        _plt.show()
        if save:
            ax.savefig(f'{path_no_title}/h.png')
        ax = _sns.catplot(data=self._df, x=self._title, kind='box', orient='h')
        ax.set_axis_labels(self._title, 'Box Plot')
        _plt.show()
        if save:
            ax.savefig(f'{path_no_title}/b.png')
        ax = _scipy.stats.probplot(self._original, dist='norm', plot=_plt)
        _plt.show()
        if save:
            ax.savefig(f'{path_no_title}/q.png')
        
    # all of the tests   
    def combine_tests(self):
        """
        This function combines all of the tests:
            1. Histogram
            2. Box Plot
            3. QQ Plot
            4. Shapiro Wilk test
            5. Kolmogorov Smirnov test
            6. Pearson's test
            7. Skew test
            8. Kurtosis test

        Parameters
        ----------
        None
        
        Returns
        -------
        DataFrame 
        """
        
        n_stat = self.normality_tests()
        p_stat = self.param_tests()
        comb = _pd.concat([n_stat, p_stat], axis=1)
        self.diag_tests()
        return comb

    
        

        
# anova
class ANOVA:
    """
    Класс для проведения однофакторного дисперсионного анализа.

    ...

    Init Attributes
    ----------
    df : pandas.DataFrame
        датафрейм
    factor : str
        название столбца факторной (категориальной) переменной 
    result : str
        название столбца результирующей (непрерывной) переменной

    Methods
    -------
    plot_diag(self, palette='pastel', save=False, path=None):
        Строит BoxPlot по входным данным.
    check_homogeneity():
        Проверяет предположение о гомогенности дисперсий.
    anova_results():
        Выводит результаты анализа.
    thsd_results():
        Выводит результаты множественных сравнений в ANOVA.
    """
    
    def __cut(self):
        self._df = self._df[[self._factor, self._result]]
    
    
    def __init__(self, df, factor, result):
        nfactor = re.sub(r'[^а-яА-Яa-zA-Z\d\s]','',factor).replace(' ','_')
        nresult = re.sub(r'[^а-яА-Яa-zA-Z\d\s]','',result).replace(' ','_')
        df = df.rename(columns={factor:nfactor,result:nresult})
        self._df = df
        self._shape = df.shape
        self._nrows = df.shape[0]
        self._factor = nfactor
        self._result = nresult
        self._model = f'{self._result} ~ C({self._factor})'
        self.__cut()
        
    # boxplot    
    def plot_diag(self, palette='pastel', save=False, path=None, order=None):
        """
        Строит BoxPlot по входным данным.

        Parameters
        ----------
        palette : str, default 'pastel'
            Палитра для графиков.
        save : bool, default False
            Сохранение графиков.
        path : str, default ``None``
            Путь до файла сохранения.
        
        Returns
        -------
        None
        """
        
        self._df[self._factor] = _pd.Categorical(self._df[self._factor],categories=order,ordered=True)
        
        gr = self._df.groupby(self._factor)[self._result]
        
        means = gr.mean()
        std_errors = gr.sem()

        _plt.figure(figsize=(11.7,8.27))
        _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)

        ax = _sns.barplot(x=means.index, y=means.values, order=order)
        ax.errorbar(x=_np.arange(len(means)), y=means.values, yerr=std_errors, fmt='none', c='black', capsize=4)

        _plt.plot(_np.arange(len(means)), means.values, marker='o', linestyle='-', color='black', linewidth=1.5, markersize=6)
            
        for i, (mean, std_error) in enumerate(zip(means.values, std_errors.values)):
            _plt.text(i+0.007, mean + 0.9, f'{mean:.2f} ± {std_error:.2f}', ha='center')
            
        _plt.title(f"{self._factor.replace('_',' ')} / {self._result.replace('_',' ')}")
        _plt.xlabel('Группы')
        _plt.ylabel(self._result.capitalize().replace('_',' '))
        _plt.show();
        
        
        if save:
            ax.savefig(path)
            
    # проверка равенства дисперсий 
    def check_homogeneity(self):
        """
        Проверяет предположение о гомогенности (однородности) дисперсий по критериям:
            1. Критерий Левене.
            2. Критерий Бартлетта

        Parameters
        ----------
        None
        
        Returns
        -------
        pandas.DataFrame 
        """
        
        obj = _stat()        
        obj.levene(df=self._df, res_var=self._result, xfac_var=self._factor)
        cur_levene = obj.levene_summary
        cur_levene.rename(columns={'Value': 'Value (Levene)'}, inplace=True)        
        obj.bartlett(df=self._df, res_var=self._result, xfac_var=self._factor)
        cur_bartlett = obj.bartlett_summary
        cur_bartlett.drop(['Parameter'], axis=1, inplace=True)
        cur_bartlett.rename(columns={'Value': 'Value (Bartlett)'}, inplace=True)                    
        final_cur = _pd.concat([cur_levene, cur_bartlett], axis=1)
        final_cur.iloc[0,0] = 'Test statistics (W/T)'            
        final_cur = _pd.concat([final_cur, _pd.DataFrame({'Parameter':['Check'], 'Value (Levene)':['True' if final_cur.iloc[2,1] > 0.05 else 'False'], 'Value (Bartlett)':['True' if final_cur.iloc[2,2] > 0.05 else 'False']})], ignore_index = True)    
        return final_cur
    
    # результаты
    def get_anova_results(self):
        """
        Выводит результаты однофакторного дисперсионного анализа.

        Parameters
        ----------
        None
        
        Returns
        -------
        pandas.DataFrame 
        """
        
        obj = _stat()
        obj.anova_stat(df=self._df, res_var=self._result, anova_model=self._model)
        cur = obj.anova_summary
        cur['Check'] = ['True' if cur.iloc[0,4] < 0.05 else 'False', '']        
        return cur
    
    # множественные сравенения
    def get_thsd_results(self):
        """
        Выводит результаты множественных сравнений в ANOVA.

        Parameters
        ----------
        None
        
        Returns
        -------
        pandas.DataFrame 
        """
        
        obj = _stat()
        obj.tukey_hsd(df=self._df, res_var=self._result, xfac_var=self._factor, anova_model=self._model)
        cur = obj.tukey_summary
        for i in range(len(cur)):
            cur.loc[i,'Check'] = 'True' if cur.loc[i,'p-value'] < 0.05 else 'False'
            cur.loc[i,'ps'] = f'p({cur.iloc[i,0]},{cur.iloc[i,1]}) = '
        cur = cur[['group1','group2','Diff','Lower','Upper','q-value','ps','p-value','Check']]           
        return cur

    
    
    
    
# Student's t-tests
class Ttests:
    # t table: https://www.tdistributiontable.com/
    
    def __init__(self, df):
        self._df = df
    
    
    def __plot_diag(self, group1, group2, gr, xt, yt, title, palette, save, path, dpi, hatch, hatch_color, hatches, context):

        if hatches == None:
            hatches = ['\\\\', 'x', '+', '-', '*', 'o']
        means = gr.mean()
        std_errors = gr.sem()

        _plt.figure(figsize=(11.7,8.27), dpi=dpi)
        _sns.set_theme(style="whitegrid", palette=palette, context=context, font_scale=1.3, color_codes=True)

        ax = _sns.barplot(x=means.index, y=means.values)
        ax.errorbar(x=_np.arange(len(means)), y=means.values, yerr=std_errors, fmt='none', c='black', capsize=4)

        _plt.plot(_np.arange(len(means)), means.values, marker='o', linestyle='-', color='black', linewidth=1.5, markersize=6)
            
        for i, (mean, std_error) in enumerate(zip(means.values, std_errors.values)):
            _plt.text(i+0.007, mean + 0.5*std_error, f'{mean:.2f}  ±  {std_error:.2f}', ha='center', c='black')
        
        if hatch:
            for i, thisbar in enumerate(ax.patches):
                thisbar.set_hatch(hatches[i])
                thisbar.set_edgecolor(hatch_color)


        if xt==None:
            xt = group1
        if yt==None:
            yt = group2
        if title==None:
            title = f'{yt} / {xt}'
            
        _plt.title(title)
        _plt.xlabel(xt)
        _plt.ylabel(yt)
        
        # _plt.legend()
        _plt.tight_layout() 
        _plt.show();
                
        
        if save:
            _plt.savefig(path)
    
    
    def __check_homogeneity(self, df, result, factor):
        obj = _stat()        
        obj.levene(df=df, res_var=result, xfac_var=factor)
        cur_levene = obj.levene_summary 
        return cur_levene
    
    # одновыборочный
    def one_sample_ttest(self, result, mu):
        cur_df = self._df[result].to_frame()
        res = _stat()
        res.ttest(df=cur_df, res=result, mu=mu, test_type=1)
        summary = res.summary
        t, pts = _scipy.stats.ttest_1samp(a=cur_df[result], popmean=mu)
        # среднее выборки > среднего ГС
        t, pg = _scipy.stats.ttest_1samp(a=cur_df[result], popmean=mu, alternative='greater')
        # среднее выборки < среднего ГС
        t, pl = _scipy.stats.ttest_1samp(a=cur_df[result], popmean=mu, alternative='less')
        res_df = _pd.DataFrame(data=list(zip(['Test statistics (T)', 'P-value (two-tail)', 'P-value (one-tail right side)', 'P-value (one-tail left side)'], [t, pts, pg, pl])), 
                              columns=['Parameter', 'Value'])
        res_df['Check'] = ['', 'True' if res_df.loc[1,'Value'] < 0.05 else 'False',
                          'True' if res_df.loc[2,'Value'] < 0.05 else 'False', 'True' if res_df.loc[3,'Value'] < 0.05 else 'False']
        print(res_df)
        print(summary)
        
    # двухвыборочный    
    def two_sample_ttest(self, group1, group2, xt=None, yt=None, title=None, grouped=True, plot=False, 
                         palette='Paired', save=False, path=None, dpi=500, hatch=False, hatch_color='black', hatches=None, context='notebook'):
        
        gg1 = group1
        gg2 = group2
        
        if grouped:
            gr = self._df.groupby(group1)[group2]
            r = gr.apply(list).apply(_pd.Series).T
            group1, group2 = r.columns
            dd = r.copy()
        else:
            dd = self._df.copy()
            new_data = []

            for index, row in dd[[group1, group2]].T.iterrows():
                for value in row:
                    new_data.append([value, index])

            new_df = _pd.DataFrame(new_data, columns=[group1, group2]).dropna().reset_index(drop=True)
            gr = new_df.groupby(group2)[group1]
        
        res = _stat()
        df_melt = _pd.melt(dd.reset_index(), id_vars=['index'], value_vars=[group1, group2], var_name='group')      
        levene = self.__check_homogeneity(df_melt, 'value', 'group')
        levene_pvalue = levene.loc[2,'Value']     
        
        eq_var = True if levene_pvalue > 0.05 else False
        
        t, pts = _scipy.stats.ttest_ind(a=dd[group1], b=dd[group2], equal_var=eq_var)
        t, pg = _scipy.stats.ttest_ind(a=dd[group1], b=dd[group2], alternative='greater', equal_var=eq_var)
        t, pl = _scipy.stats.ttest_ind(a=dd[group1], b=dd[group2], alternative='less', equal_var=eq_var)
        
        res_df = _pd.DataFrame(data=list(zip(['Test statistics (T)', 'P-value (two-tail)', 'P-value (one-tail right side)', 'P-value (one-tail left side)'], [t, pts, pg, pl])), 
                                columns=['Parameter', 'Value'])
        res_df['Check'] = ['', 'True' if res_df.loc[1,'Value'] < 0.05 else 'False',
                            'True' if res_df.loc[2,'Value'] < 0.05 else 'False', 'True' if res_df.loc[3,'Value'] < 0.05 else 'False']
        res.ttest(df=df_melt, xfac='group', res='value', evar=eq_var, test_type=2)
        
        if plot:
            self.__plot_diag(gg1, gg2, gr, xt, yt, title, palette, save, path, dpi, hatch, hatch_color, hatches, context)
              
        print(res_df)
        print(res.summary)
        
    # двухвыборочный для зависимых выборок (парный)
    def paired_ttest(self, group1, group2, grouped=True, plot=False, palette='pastel', save=False, path=None):
        
        if grouped:
            gr = self._df.groupby(group1)[group2]
            r = gr.apply(list).apply(_pd.Series).T
            group1, group2 = r.columns
            dd = r.copy()
        else:
            dd = self._df.copy()
            new_data = []

            for index, row in dd[[group1, group2]].T.iterrows():
                for value in row:
                    new_data.append([value, index])

            new_df = _pd.DataFrame(new_data, columns=[group1, group2]).dropna().reset_index(drop=True)
            gr = new_df.groupby(group2)[group1]
        
        res = _stat()
        cur_df = dd[[group1, group2]]
        res.ttest(df=cur_df, res=[group1, group2], test_type=3)
        if plot:
            pass
            self.__plot_diag(group1, group2, gr, palette, save, path)  
        print(res.summary)
        


class Correlation:
    
    def get_results(self, x, y, type='P'):
        if type == 'P':
            corr, pv = _scipy.stats.pearsonr(x, y)
        elif type == 'S':
            corr, pv = _scipy.stats.spearmanr(x, y)
        res_df = _pd.DataFrame(data=list(zip(['Коэффициент корреляции', 'p-значение'], [corr, pv])), 
                                columns=['Parameter', 'Value'])
        res_df['Check'] = ['', 'True' if res_df.loc[1,'Value'] < 0.05 else 'False']
        return res_df



class Kruskal:
    
    def __init__(self, df) -> None:
        self.df = df
        
    def _wrap_labels(self, ax, width, break_long_words=False):
        labels = []
        for axi in ax.axes.flat:
            labels = [label.get_text() for label in axi.get_xticklabels()]
            wrapped_labels = [_tp.fill(text, width=width, break_long_words=break_long_words) for text in labels]
            ax.set_xticklabels(wrapped_labels, rotation=0)
        
    def get_kruskal_results(self, *samples):
        self.ss = []
        for i in samples:
            self.ss.append(self.df[i])
        
        stat, p = _scipy.stats.kruskal(*self.ss, nan_policy='omit')
        self._df_melt1 = _pd.DataFrame([*self.ss]).T
        # print(self._df_melt1)
        self._df_melt = _pd.melt(self._df_melt1, value_vars=self._df_melt1.columns,
                                 var_name='Группа', value_name='Значение').dropna().reset_index(drop=True)
        # print(self._df_melt)
        res = _pd.DataFrame(data=list(zip(['Test statistics (H)', 'p value', 'Check'], [stat, p, p])), columns=['Parameter', 'Value'])
        res.loc[2,'Value'] = True if res.loc[1,'Value'] < 0.05 else False
        return res
    
    def get_dunn_results(self, dd, val, group):
        dunn_results = _sp.posthoc_dunn(dd, val_col=val, group_col=group, p_adjust='holm')
        return dunn_results
    
        
    def plot_diag(self, palette='vlag', xt='Группа', yt='', ttl='', order=None):
        
        
        self._df_melt['Группа'] = _pd.Categorical(self._df_melt['Группа'],categories=order,ordered=True)
        gr = self._df_melt.groupby('Группа')['Значение']
        
        _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.05)
        ax = _sns.catplot(data=self._df_melt, x='Группа', y='Значение', height=8.27, aspect=11.7/8.27, kind='box', order=order)
        
        # gr = self._df_melt.groupby('Группа')['Значение']
        meds = gr.median()
        # print(meds)
        qq = gr.quantile(q=[0.25,0.75])
        
        _plt.plot(_np.arange(len(meds)), meds.values, marker='o', linestyle='-', color='black', linewidth=1.2, markersize=6)
        
        for i, (mean, qq) in enumerate(zip(meds.values, qq.values.reshape((self.df.shape[1], 2)))):
            tt = f'{mean:.1f} [{qq[0]:.1f},{qq[1]:.1f}]'
            if mean < qq[0]+(qq[1]-qq[0])*0.5:
                _plt.text(i+0.02, mean+(qq[1]-qq[0])*0.25, tt, ha='center', c='black')
            else:
                _plt.text(i+0.02, mean-(qq[1]-qq[0])*0.25, tt, ha='center', c='black')
            print(tt)
            
        self._wrap_labels(ax, 80/meds.shape[0])
        
        # for i, mean in enumerate(meds.values):
        #     _plt.text(i+0.007, mean+3, f'{mean:.2f}', ha='center')
        
        _plt.xlabel(xt)
        _plt.ylabel(yt)
        if ttl:
            _plt.title(ttl)
        else:
            _plt.title(f'{yt} / {xt}')
        _plt.show();              
        
        
        
# Mann-Whitney U test
class RankTests2S:
    """
    Class for rank tests

    ...

    Init Attributes
    ----------
    df : DataFrame
        data frame
    group1 : Series
        first sample
    group2 : Series
        second sample

    Methods
    -------
    
    get_mwu_results(self, alt='two-sided'):
        Perform the Mann-Whitney U rank test on two independent samples
    plot_diag(self, palette='pastel', save=False, path='', lang='eng'):
        Visualize catplot for two samples
            
    """
    
    def __init__(self, df, group1, group2):
        self._df = df
        self._shape = df.shape
        self._nrows = df.shape[0]
        self._group1 = group1
        self._group2 = group2
        new_data = []
        for index, row in df[[group1, group2]].T.iterrows():
            for value in row:
                new_data.append([value, index])
        self._df_melt = _pd.DataFrame(new_data, columns=[group1, group2]).dropna().reset_index(drop=True)


    def _wrap_labels(self, ax, width, break_long_words=False):
        labels = []
        for label in ax.get_xticklabels():
            text = label.get_text()
            labels.append(_tp.fill(text, width=width,
                            break_long_words=break_long_words))
        ax.set_xticklabels(labels, rotation=0)

        
    def plot_diag(self, dpi=500, hatch=False, hatch_color='black', hatches=None, palette='Paired',
                  save=False, path='', lang='eng', xt=None, yt=None, title=None, box_names=None, context='notebook'):
        """
        Visualize catplot for two samples

        Parameters
        ----------
        palette : str, default 'pastel'
            palette for visualizations
        save : bool, default False
            plots saving
        path : str, default ``None``
            path to the save folder
        lang : str, default 'eng'
            axis labels language
        
        Returns
        -------
        None
        """
        
        _plt.figure(figsize=(11.7,8.27), dpi=dpi)
        _sns.set_theme(style="whitegrid", palette=palette, context=context, font_scale=1.3)
        ax = _sns.boxplot(data=self._df_melt, x=self._group2, y=self._group1)
        
        gr = self._df_melt.groupby(self._group2)[self._group1]
        
        if hatches == None:
            hatches = ['\\\\', 'x', '+', '-', '*', 'o']
        
        meds = gr.median()
        qq = gr.quantile(q=[0.25,0.75])
        
        _plt.plot(_np.arange(len(meds)), meds.values, marker='o', linestyle='-', color='black', linewidth=1.2, markersize=6)
            
        for i, (mean, qq) in enumerate(zip(meds.values, qq.values.reshape((2, 2)))):
            tt = f'{mean:.1f} [{qq[0]:.1f},{qq[1]:.1f}]'
            _plt.text(i+0.02, mean+(qq[1]-qq[0])*0.25, tt, ha='center', c='black')
            print(tt)
            
        if hatch:
            for i, artist in enumerate(ax.patches):
                artist.set_hatch(hatches[i % len(hatches)]) 
                artist.set_edgecolor(hatch_color)
       
        self._wrap_labels(ax, 80/meds.shape[0])
        
        if box_names:
            ax.set_xticklabels(box_names)
        
        
        if xt==None:
            xt = self._group1
        if yt==None:
            yt = self._group2
        if title==None:
            title = f'{yt} / {xt}'
            
        _plt.xlabel(xt)
        _plt.ylabel(yt)
        _plt.title(title)
        
        _plt.tight_layout() 
        _plt.show()
        
        if save:
            _plt.savefig(path)
    
    
    def get_mwu_results(self, alt='two-sided'):
        """
        Perform the Mann-Whitney U rank test on two independent samples

        Parameters
        ----------
        alt : str, default 'two-sided'
            defines the alternative hypothesis:
            - two-sided
            - less
            - greater
        
        Returns
        -------
        DataFrame
        """
        
        stat, p = _scipy.stats.mannwhitneyu(x=self._df_melt[self._df_melt[self._group2]==self._group2][self._group1].values,
                                            y=self._df_melt[self._df_melt[self._group2]==self._group1][self._group1].values, 
                                            alternative=alt)
        res = _pd.DataFrame(data=list(zip(['Test statistics (U)', 'p value', 'Check'], [stat, p, p])), columns=['Parameter', 'Value'])
        res.loc[2,'Value'] = True if res.loc[1,'Value'] < 0.05 else False
        return res


    def get_wilcoxon_results(self, alt='two-sided', correction=True):
        """
        Calculate the Wilcoxon signed-rank test

        Parameters
        ----------
        alt : str, default 'two-sided'
            defines the alternative hypothesis:
            - two-sided
            - less
            - greater
        correction : bool, default True
            apply continuity correction
            
        Returns
        -------
        DataFrame
        """
        
        stat, p = _scipy.stats.wilcoxon(x=self._df[self._group1],
                                        y=self._df[self._group2], 
                                        alternative=alt, correction=correction)
        res = _pd.DataFrame(data=list(zip(['Test statistics (W)', 'p value', 'Check'], [stat, p, p])), columns=['Parameter', 'Value'])
        res.loc[2,'Value'] = True if res.loc[1,'Value'] < 0.05 else False
        return res
    
    
    
    
# тесты для номинальных данных
class CrosstTests:
    _nround = 6
    
    def _wrap_labels(self, ax, width, break_long_words=False):
        labels = []
        for label in ax.get_xticklabels():
            text = label.get_text()
            labels.append(_tp.fill(text, width=width,
                            break_long_words=break_long_words))
        ax.set_xticklabels(labels, rotation=0)
    
    # обработка отсутствия данных на пересечениях
    def __proc_zeros(self, temp):
        for j in (self._original[self._group1].unique()):
            for i in (self._original[self._group2].unique()):
                cur_mindex = (i, j)
                if all(cur_mindex != item for item in temp.index.values):
                    temp[cur_mindex] = 0
        return temp
    
    # сумма рядов, столбцов, тотал (Фишер)
    def __get_sums(self):
        row_sum = []
        col_sum = []
        for i in range(self._nrows):
            temp = 0
            for j in range(self._ncols):
                temp += self._df[i][j]
            row_sum.append(temp)
        for j in range(self._ncols):
            temp = 0
            for i in range(self._nrows):
                temp += self._df[i][j]
            col_sum.append(temp)
        self._row_sum = row_sum
        self._col_sum = col_sum
        self._total = _np.array(row_sum).sum()
    
    
    def __init__(self, data, group1, group2):
        self._original = data
        self._group1 = group1
        self._group2 = group2
        self._nrows = data[group1].unique().shape[0]
        self._ncols = data[group2].unique().shape[0]
        temp = data.groupby(group2)[group1].value_counts().sort_index(level=1)
        temp = self.__proc_zeros(temp)
        self._df = temp.sort_index(level=1).to_numpy().reshape(self._nrows,self._ncols)
        self._shape = self._df.shape
        self.__get_sums()
        
    # перестановки (Фишер)   
    def __dfs(self, mat, pos, r_sum, c_sum, p_0, p):
        (xx, yy) = pos
        (r, c) = (len(r_sum), len(c_sum))
        mat_new = []
        for i in range(len(mat)):
            temp = []
            for j in range(len(mat[0])):
                temp.append(mat[i][j])
            mat_new.append(temp)
        if xx == -1 and yy == -1:
            for i in range(r-1):
                temp = r_sum[i]
                for j in range(c-1):
                    temp -= mat_new[i][j]
                mat_new[i][c-1] = temp
            for j in range(c-1):
                temp = c_sum[j]
                for i in range(r-1):
                    temp -= mat_new[i][j]
                mat_new[r-1][j] = temp
            temp = r_sum[r-1]
            for j in range(c-1):
                temp -= mat_new[r-1][j]
            if temp <0:
                return
            mat_new[r-1][c-1] = temp
            p_1 = 1
            for x in r_sum:
                p_1 *= _math.factorial(x)
            for y in c_sum:
                p_1 *= _math.factorial(y)
            n = 0
            for x in r_sum:
                n += x
            p_1 /= _math.factorial(n)
            for i in range(len(mat_new)):
                for j in range(len(mat_new[0])):
                    p_1 /= _math.factorial(mat_new[i][j])
            if p_1 <= p_0 + 0.00000001:
                p[0] += p_1
        else:
            max_1 = r_sum[xx]
            max_2 = c_sum[yy]
            for j in range(c):
                max_1 -= mat_new[xx][j]
            for i in range(r):
                max_2 -= mat_new[i][yy]
            for k in range(min(max_1,max_2)+1):
                mat_new[xx][yy] = k
                if xx == r-2 and yy == c-2:
                    pos_new = (-1, -1)
                elif xx == r-2:
                    pos_new = (0, yy+1)
                else:
                    pos_new = (xx+1, yy)
                self.__dfs(mat_new, pos_new, r_sum, c_sum, p_0, p)
        
    # первоначальное p для текущей таблицы  
    def __fisher_exact_p(self):
        mat = [[0] * len(self._col_sum)] * len(self._row_sum)
        pos = (0, 0)
        p_0 = 1
        n = 0
        for x in self._row_sum:
            p_0 *= _math.factorial(x)
            n += x
        for y in self._col_sum:
            p_0 *= _math.factorial(y)            
        p_0 /= _math.factorial(n)
        for i in range(self._nrows):
            for j in range(self._ncols):
                p_0 /= _math.factorial(self._df[i][j])
        p = [0]
        self.__dfs(mat, pos, self._row_sum, self._col_sum, p_0, p)
        return p[0]
    
    # таблица сопряженности
    def get_contingency_table(self, norm=True, palette='Paired', plot_pie=False, plot_mosaic=False, default_mosaic=True, 
                              plot_count=False, save=False, path_no_title=None, order=None, hue_order=None,
                              xt=None, yt='Количество', title=None, dpi=500, hatch=False,
                              hatches=None, hatch_color='black', legend_title=None, 
                              legend_pos='upper right'):
        # custom _pd.crosstab()
        unique_g1 = _np.sort(self._original[self._group1].unique(), axis=None)
        unique_g2 = _np.sort(self._original[self._group2].unique(), axis=None)
        uu1 = unique_g1.copy()
        uu2 = unique_g2.copy()
        unique_g1 = unique_g1.astype('object')
        for i in range(len(unique_g1)):
            unique_g1[i] = f'{self._group1} - {unique_g1[i]}'
        unique_g2 = unique_g2.astype('object')
        for i in range(len(unique_g2)):
            unique_g2[i] = f'{self._group2} - {unique_g2[i]}'
        frame = _pd.DataFrame(data=self._df, columns=unique_g2, index=unique_g1)        
        frame['Total'] = self._row_sum
        temp = []
        for i in self._col_sum:
            temp.append(i)
        temp.append(self._total)
        frame.loc[len(frame.index)] = temp
        frame.rename(index={len(frame.index)-1:'Total'}, inplace=True)

        
        if plot_pie:
            values2 = frame.iloc[-1,:-1].values
            values1 = frame.iloc[:-1,-1].values
            labels1 = [uu1[0], uu1[1]]
            labels2 = [uu2[0], uu2[1]]
            
            _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)
            fig, axes = _plt.subplots(1, 2, figsize=(12, 6))
            # _plt.subplots_adjust(wspace=5)

            def autopct_format(values):
                def func(pct):
                    total = sum(values)
                    val = int(round(pct*total/100.0))
                    return "{}\n({:.1f})%".format(val, pct)
                return func

            pie1 = axes[0].pie(values1, labels=labels1, autopct=autopct_format(values1), startangle=90, colors=_sns.color_palette(palette))
            axes[0].set_title(self._group1.replace(' (','\n('))
            pie2 = axes[1].pie(values2, labels=labels2, autopct=autopct_format(values2), startangle=90, colors=_sns.color_palette(palette))
            axes[1].set_title(self._group2.replace(' (','\n('))

            for text in pie1[2]:
                text.set_color('black')
            for text in pie2[2]:
                text.set_color('black')
            
            _plt.tight_layout()
            _plt.show();
        
        
        # count график
        if plot_count:
            
            _plt.figure(figsize=(11.7,8.27), dpi=dpi)
            _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)
            ax = _sns.countplot(data=self._original, x=self._group1, hue=self._group2, 
                                order=order, hue_order=hue_order)
            
            if hatches == None:
                hatches = ['\\\\', 'x', '+', '-', '*', 'o']
                h_size = self._original[self._group1].nunique()
                hatches = [val for val in hatches for _ in range(h_size)]
            if hatch:
                for i, artist in enumerate(ax.patches):
                    artist.set_hatch(hatches[i % len(hatches)]) 
                    artist.set_edgecolor(hatch_color)
                    
            self._wrap_labels(ax, 80/(h_size))
            
            if xt==None:
                xt = self._group1
            if yt==None:
                yt = self._group2
            if title==None:
                title = f'{yt} / {xt}'
                
            _plt.xlabel(xt)
            _plt.ylabel(yt)
            _plt.title(title)
            if legend_title == None:
                legend_title = self._group2
            
            _plt.legend(title=legend_title, loc=legend_pos)
            _plt.tight_layout() 
            _plt.show()

                
        # график-мозаика
        if plot_mosaic:
            mosaic_frame = frame.iloc[:-1,:-1]
            mosaic_idxs = mosaic_frame.index.values
            mosaic_cols = mosaic_frame.columns.values
            mosaic_values = mosaic_frame.values
            mosaic_dict = {}
            for i in range(self._nrows-1,-1,-1):
                for j in range(self._ncols):
                    mosaic_dict[(mosaic_cols[j], mosaic_idxs[i])] = mosaic_values[i,j]
            if default_mosaic:
                ax = _msc(mosaic_dict)
            else:
                # другой вид мозайки
                ax = _msc(self._original, [self._group1, self._group2])
            _plt.show()
            if save:
                _plt.savefig(f'{path_no_title}/mosaic.png')
   
        # добавляет нормализованные значения через crosstab()
        if norm:
            crosst = _pd.crosstab(index=self._original[self._group1], columns=self._original[self._group2], margins=True, normalize=True)
            for i in range(self._nrows+1):
                for j in range(self._ncols+1):
                    frame.iat[i,j] = f'{frame.iat[i,j]} ({round(crosst.values[i][j]*100,2)}%)'   
        return frame
    
    # точный критерий Фишера (двусторонний)
    def fisher_exact_pvalue(self):
        res = round(self.__fisher_exact_p(), self._nround)
        res_df = _pd.DataFrame(data=list(zip(['p value', 'Check'], [res, res])), columns=['Parameter', 'Value'])
        res_df.loc[1,'Value'] = True if res_df.loc[0,'Value'] < 0.05 else False
        return res_df
        
    # хи-кв. -> 80% ячеек > 5 наблюдений и ни в одной нет < 1, иначе -> точный Фишер
    # критерий Хи-квадрат
    def chi2_pvalue(self):
        chi2, p = _chi2cont(self._df)[:2]
        res_df = _pd.DataFrame(data=list(zip(['Test statistics (Chi2)', 'p value', 'Check'], [chi2, p, p])), columns=['Parameter', 'Value'])
        res_df.loc[2,'Value'] = True if res_df.loc[1,'Value'] < 0.05 else False
        return res_df
    
    def mcn(self, type=1):
        
        if type==1: # ДЕФОЛТ - тест МакНемара с поправкой Эдвардса
            result = mcnemar(self._df.T, exact=False, correction=True)
        elif type==2: # стандартный тест МакНемара
            result = mcnemar(self._df.T, exact=False, correction=False)
        elif type==3: # точный тест МакНемара (когда b+c<25)
            result = mcnemar(self._df.T, exact=True)
            
        res_df = _pd.DataFrame(data=list(zip(['Test statistics (Q)', 'p value', 'Check'], [result.statistic, result.pvalue, result.pvalue])), columns=['Parameter', 'Value'])
        res_df.loc[2,'Value'] = True if res_df.loc[1,'Value'] < 0.05 else False
        return res_df



