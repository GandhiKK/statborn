import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import seaborn as _sns
import scipy as _scipy
import warnings as _w
import math as _math 
from bioinfokit.analys import stat as _stat
from scipy.stats import chi2_contingency as _chi2cont
from statsmodels.graphics.mosaicplot import mosaic as _msc
_w.filterwarnings("ignore")




# проверка нормальности распределения
class normality:
    """
    Класс для проверки данных на нормальность.

    ...

    Init Attributes
    ----------
    df : pandas.DataFrame
        датафрейм
    col : str
        название столбца

    Methods
    -------
    normality_tests():
        Использует критерии нормальности для проверки:
        1. Критерий Шапиро-Уилка.
        2. Критерий Колмогорова-Смирнова.
        3. Критерий согласия Пирсона.
    param_tests():
        Использует критерий асимметрии и эксцесса.
    diag_tests(palette='pastel', save=False, path_no_title=None):
        Использует графические критерии:
        1. Гистограмма.
        2. BoxPlot.
        3. QQ-Plot.
    combine_tests()
        Использует все критерии.
    """
    
    _nround = 6
    
    def __init__(self, df=None, col=None):
        self._df = _pd.DataFrame(df[col])
        self._original = df[col]
        self._shape = self._df.shape
        self._nrows = self._df.shape[0]
        self._title = self._df.columns.values[0]

    # критерий Шапиро-Уилка   
    def __shapiro(self, stat):
        try:
            stat_SW, p_SW = _scipy.stats.shapiro(self._df)
        except ValueError:
            stat_SW, p_SW = _np.NaN, _np.NaN
        stat['Shapiro–Wilk'] = [round(stat_SW, self._nround), round(p_SW, self._nround), 'True' if p_SW > 0.05 else 'False']
        return stat
    
    # критерий Колмогорова-Смирнова
    def __ks(self, stat):
        try:
            stat_KS, p_KS = _scipy.stats.kstest(self._df, 'norm', args=(_np.mean(self._df), _np.std(self._df)))
        except ValueError:
            stat_KS, p_KS = _np.NaN, _np.NaN
        stat['Kolmogorov-Smirnov'] = [round(stat_KS, self._nround), round(p_KS, self._nround), 'True' if p_KS > 0.2 else 'False']
        return stat
    
    # критерий согласия Пирсона
    def __pearson(self, stat):
        try:
            stat_P, p_P = _scipy.stats.normaltest(self._original)
        except ValueError:
            stat_P, p_P = _np.NaN, _np.NaN
        stat['Pearson'] = [round(stat_P, self._nround), round(p_P, self._nround), 'True' if p_P > 0.05 else 'False']  
        return stat
    
    # тест ассиметрии
    def __skew(self, stat):
        try: 
            stat_skew, p_skew = _scipy.stats.skewtest(self._original)
        except ValueError:
            stat_skew, p_skew = _np.NaN, _np.NaN
        stat['Skew test'] = [round(stat_skew, self._nround), round(p_skew, self._nround), 'True' if p_skew > 0.05 else 'False']
        return stat
    
    # тест эксцесса
    def __kurtosis(self, stat):
        try:
            stat_kurt, p_kurt = _scipy.stats.kurtosistest(self._original)
        except ValueError:
            stat_kurt, p_kurt = _np.NaN, _np.NaN
        stat['Kurtosis test'] = [round(stat_kurt, self._nround), round(p_kurt, self._nround), 'True' if p_kurt > 0.05 else 'False']
        return stat
    
    # критерии
    def normality_tests(self):
        """
        Проводит проверку по критериям нормальности:
            1. Критерий Шапиро-Уилка.
            2. Критерий Колмогорова-Смирнова.
            3. Критерий согласия Пирсона.

        Parameters
        ----------
        None
        
        Returns
        -------
        pandas.DataFrame 
        """
        
        statistics = _pd.DataFrame(index=['Test statistics', 'p value', 'Check'])
        statistics = self.__shapiro(statistics)
        statistics = self.__ks(statistics)
        statistics = self.__pearson(statistics)        
        return statistics
    
    # ассиметрия/эксцесс
    def param_tests(self):
        """
        Проводит проверку по критериям асимметрии и эксцесса.

        Parameters
        ----------
        None
        
        Returns
        -------
        pandas.DataFrame 
        """
        
        statistics = _pd.DataFrame(index=['Test statistics', 'p value', 'Check'])
        statistics = self.__skew(statistics)
        statistics = self.__kurtosis(statistics)
        return statistics
    
    # графики
    def diag_tests(self, palette='pastel', save=False, path_no_title=None):
        """
        Проводит проверку по графическим критериям нормальности:
            1. Гистограмма.
            2. BoxPlot.
            3. QQ-Plot.

        Parameters
        ----------
        palette : str, default 'pastel'
            Палитра для графиков.
        save : bool, default False
            Сохранение графиков.
        path_no_title : str, default ``None``
            Путь до папки сохранения.
        
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
        
    # все тесты    
    def combine_tests(self):
        """
        Проводит проверку по всем доступным критериям:
            1. Гистограмма.
            2. BoxPlot.
            3. QQ-Plot.
            4. Критерий Шапиро-Уилка.
            5. Критерий Колмогорова-Смирнова.
            6. Критерий согласия Пирсона.
            7. Критерий асимметрии.
            8. Критерий эксцесса.

        Parameters
        ----------
        None
        
        Returns
        -------
        pandas.DataFrame 
        """
        
        n_stat = self.normality_tests()
        p_stat = self.param_tests()
        comb = _pd.concat([n_stat, p_stat], axis=1)
        self.diag_tests()
        return comb

    
        

        
# однофакторный дисперсионный анализ
class anova:
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
        self._df = df
        self._shape = df.shape
        self._nrows = df.shape[0]
        self._factor = factor
        self._result = result
        self._model = f'{result} ~ C({factor})'
        self.__cut()
        
    # boxplot    
    def plot_diag(self, palette='pastel', save=False, path=None):
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
        
        _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)
        ax = _sns.catplot(x=self._factor, y=self._result, data=self._df, 
                         height=8.27, aspect=11.7/8.27, kind='box')
        ax.set_axis_labels('Группы', self._result.capitalize())
        _plt.show()
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
    def anova_results(self):
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
    def thsd_results(self):
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

    
    
    
    
# Т критерий Стьюдента
class t_tests:
    # t table: https://www.tdistributiontable.com/
    def __init__(self, df):
        self._df = df
    
    
    def __plot_diag(self, group1, group2, palette, save, path):
        _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)
        df_melt = _pd.melt(self._df.reset_index(), id_vars=['index'], value_vars=[group1, group2], var_name='group')
        ax = _sns.catplot(x='group', y='value', data=df_melt, height=8.27, aspect=11.7/8.27, kind='box')
        ax.set_axis_labels('Группы', 'Значение')
        _plt.show()
        if save:
            ax.savefig(path)
    
    
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
    def two_sample_ttest(self, group1, group2, plot=False, palette='pastel', save=False, path=None):
        res = _stat()
        df_melt = _pd.melt(self._df.reset_index(), id_vars=['index'], value_vars=[group1, group2], var_name='group')      
        levene = self.__check_homogeneity(df_melt, 'value', 'group')
        levene_pvalue = levene.loc[2,'Value']     
        if levene_pvalue > 0.05:
            t, pts = _scipy.stats.ttest_ind(a=self._df[group1], b=self._df[group2])
            t, pg = _scipy.stats.ttest_ind(a=self._df[group1], b=self._df[group2], alternative='greater')
            t, pl = _scipy.stats.ttest_ind(a=self._df[group1], b=self._df[group2], alternative='less')
            res_df = _pd.DataFrame(data=list(zip(['Test statistics (T)', 'P-value (two-tail)', 'P-value (one-tail right side)', 'P-value (one-tail left side)'], [t, pts, pg, pl])), 
                                  columns=['Parameter', 'Value'])
            res_df['Check'] = ['', 'True' if res_df.loc[1,'Value'] < 0.05 else 'False',
                              'True' if res_df.loc[2,'Value'] < 0.05 else 'False', 'True' if res_df.loc[3,'Value'] < 0.05 else 'False']
            res.ttest(df=df_melt, xfac='group', res='value', test_type=2)
        else:
            t, pts = _scipy.stats.ttest_ind(a=self._df[group1], b=self._df[group2], equal_var=False)
            t, pg = _scipy.stats.ttest_ind(a=self._df[group1], b=self._df[group2], alternative='greater', equal_var=False)
            t, pl = _scipy.stats.ttest_ind(a=self._df[group1], b=self._df[group2], alternative='less', equal_var=False)
            res_df = _pd.DataFrame(data=list(zip(['Test statistics (T)', 'P-value (two-tail)', 'P-value (one-tail right side)', 'P-value (one-tail left side)'], [t, pts, pg, pl])), 
                                  columns=['Parameter', 'Value'])
            res_df['Check'] = ['', 'True' if res_df.loc[1,'Value'] < 0.05 else 'False',
                              'True' if res_df.loc[2,'Value'] < 0.05 else 'False', 'True' if res_df.loc[3,'Value'] < 0.05 else 'False']
            res.ttest(df=df_melt, xfac='group', res='value', evar=False, test_type=2)
        
        if plot:
            self.__plot_diag(group1, group2, palette, save, path)      
        print(res_df)
        print(res.summary)
        
    # двухвыборочный для зависимых выборок (парный)
    def paired_ttest(self, groupA, groupB, plot=False, palette='pastel', save=False, path=None):
        res = _stat()
        cur_df = self._df[[groupA, groupB]]
        res.ttest(df=cur_df, res=[groupA, groupB], test_type=3)
        if plot:
            self.__plot_diag(groupA, groupB, palette, save, path)  
        print(res.summary)
        

        
                
        
# U критерий Манна-Уитни (default alt = двусторонний, или задать односторонний (greater, less))
class MW_Utest:
    def __init__(self, df, group1, group2):
        self._df = df
        self._shape = df.shape
        self._nrows = df.shape[0]
        self._group1 = group1
        self._group2 = group2

        
    def plot_diag(self, palette='pastel', save=False, path=''):
        _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)
        df_melt = _pd.melt(self._df.reset_index(), id_vars=['index'], value_vars=[self._group1, self._group2], var_name='group')
        ax = _sns.catplot(x='group', y='value', data=df_melt, height=8.27, aspect=11.7/8.27, kind='box')
        ax.set_axis_labels('Группы', 'Значение')
        _plt.show()
        if save:
            ax.savefig(path)
    
    
    def MW_results(self, alt='two-sided'):
        stat, p = _scipy.stats.mannwhitneyu(x=self._df[self._group1], y=self._df[self._group2], alternative=alt)
        res = _pd.DataFrame(data=list(zip(['Test statistics (U)', 'p value', 'Check'], [stat, p, p])), columns=['Parameter', 'Value'])
        res.loc[2,'Value'] = True if res.loc[1,'Value'] < 0.05 else False
        return res

    
    
    
    
# тесты для номинальных данных
class crosst_tests:
    _nround = 6
    
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
    def get_contingency_table(self, norm=True, palette='pastel', plot_mosaic=False, default_mosaic=True, plot_count=False, save=False, path_no_title=None, order=None, hue_order=None):
        # custom _pd.crosstab()
        unique_g1 = _np.sort(self._original[self._group1].unique(), axis=None)
        unique_g2 = _np.sort(self._original[self._group2].unique(), axis=None)
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
        
        # count график
        if plot_count:
            _sns.set_theme(style="whitegrid", palette=palette, context='notebook', font_scale=1.3)
            ax = _sns.catplot(data=self._original, x=self._group1, hue=self._group2, 
                              kind='count', height=8.27, aspect=11.7/8.27, order=order, hue_order=hue_order)  
            ax.set_axis_labels(self._group1, 'Количество')
            _plt.show()
            if save:
                ax.savefig(f'{path_no_title}/count.png')
            return
                
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
                ax.savefig(f'{path_no_title}/mosaic.png')
            return
   
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



