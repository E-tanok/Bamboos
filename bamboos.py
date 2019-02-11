class Bamboos():
    import numpy as np

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.orig_columns = list(self.dataframe.columns)
        self.n_orig_rows = dataframe.shape[0]
        self.n_orig_cols = dataframe.shape[1]
        self.cols = list(dataframe.columns)
        self.new_cols = []
        self.deleted_cols = []
        self.n_rows = None
        self.n_cols = None
        self.step_id = 0
        self.step_label = 'init'

        self.possible_keys = []

        #self.df_datatypes = pd.DataFrame(columns = ['step_id', 'step_label', 'column', 'datatype'])
        self.cols_caracs = pd.DataFrame(columns = ['step_id', 'step_label', 'col', 'col_datatype', 'completion_ratio', 'unique_ratio'])

        self.orig_cols_caracs = self.get_cols_caracs()
        self.orig_incomplete_cols = self.get_incomplete_cols()

        self.L_metadata_labels = ['step_id', 'step_label', 'global_completion_ratio',
                                  'incomplete_cols_ratio' ,'n_rows', 'n_cols', 'cols',
                                  'n_incomplete_cols', 'incomplete_cols', 'n_new_cols',
                                  'new_cols', 'n_deleted_cols','deleted_cols']
        self.metadata_buffer = pd.DataFrame(columns=self.L_metadata_labels)
        self.metadata_buffer = self.update_metadata_buffer(dataframe)


    def get_cols_caracs(self, arg='last'):
        dict_col_caracs = {}
        dataframe_length = len(self.dataframe)

        if len(self.dataframe)>0:
            for col in self.dataframe.columns:
                dict_col_caracs[col] = {}
                col_datatype = str(self.dataframe[col].dtype)
                dict_col_caracs[col]['step_id'] = self.step_id
                dict_col_caracs[col]['step_label'] = self.step_label
                dict_col_caracs[col]['col'] = col
                dict_col_caracs[col]['col_datatype'] = col_datatype

                completion_ratio = len(self.dataframe[col][~self.dataframe[col].isnull()])/dataframe_length
                unique_ratio = len(np.unique(list(self.dataframe[col])))/dataframe_length
                dict_col_caracs[col]['completion_ratio'] = completion_ratio
                dict_col_caracs[col]['unique_ratio'] = unique_ratio
                dict_col_caracs[col]['step_label'] = self.step_label
                if unique_ratio == 1.0:
                    self.possible_keys.append(col)

            self.cols_caracs = self.cols_caracs.append(pd.DataFrame(dict_col_caracs).T)
            self.cols_caracs.drop_duplicates(subset=['step_id', 'col'], inplace=True)
            self.cols_caracs = self.cols_caracs[['step_id', 'step_label', 'col_datatype', 'completion_ratio', 'unique_ratio']]
            if arg == 'last':
                last_cols_caracs = self.cols_caracs[self.cols_caracs['step_id']==np.max(self.cols_caracs['step_id'])]
                return last_cols_caracs
            else:
                return self.cols_caracs
        else:
            return 'empty df !'

    def get_incomplete_cols(self):
        df_incomplete_cols = self.cols_caracs[(self.cols_caracs['completion_ratio']<1.0)&(self.cols_caracs['step_label']==self.step_label)]
        df_incomplete_cols.sort_values(by='completion_ratio', ascending=False, inplace=True)
        self.incomplete_cols = list(df_incomplete_cols.index)
        self.n_incomplete_cols = len(self.incomplete_cols)
        return self.incomplete_cols

    def flag_one_step(self, flag_label, dataframe):
        print("flagging : %s"%flag_label)
        self.step_id+=1
        self.step_label = flag_label
        #self.get_cols_caracs()
        self.update_metadata_buffer(dataframe)

    def update_metadata_buffer(self, dataframe):
        self.new_cols = [col for col in dataframe.columns if not col in self.cols]
        self.n_new_cols = len(self.new_cols)
        self.deleted_cols = [col for col in self.cols if not col in dataframe.columns]
        self.n_deleted_cols = len(self.deleted_cols)
        self.dataframe = dataframe
        self.get_cols_caracs()
        self.get_incomplete_cols()
        self.cols = list(self.dataframe.columns)
        self.n_rows = dataframe.shape[0]
        self.n_cols = dataframe.shape[1]
        self.global_completion_ratio = np.sum(self.cols_caracs['completion_ratio'])/len(self.cols_caracs['completion_ratio'])
        self.incomplete_cols_ratio = len(set(self.incomplete_cols).intersection(set(self.cols)))/len(self.cols)
        self.L_metadata_values = [self.step_id, self.step_label, self.global_completion_ratio,
                                  self.incomplete_cols_ratio, self.n_rows, self.n_cols, self.cols,
                                  self.n_incomplete_cols, self.incomplete_cols, self.n_new_cols,
                                  self.new_cols, self.n_deleted_cols, self.deleted_cols]
        dict_metadata_buffer = {label:value for label,value in zip(self.L_metadata_labels, self.L_metadata_values)}
        self.metadata_buffer = self.metadata_buffer.append(dict_metadata_buffer, ignore_index=True)

        cols_with_list_type = ['cols', 'incomplete_cols', 'new_cols', 'deleted_cols']
        for col in cols_with_list_type:
            self.metadata_buffer[col] = self.metadata_buffer[col].astype(str)

        self.metadata_buffer.drop_duplicates(inplace=True)
        self.metadata_buffer.sort_values(by='step_id', ascending=False, inplace=True)

        for col in cols_with_list_type:
            self.metadata_buffer = self.convert_in_list(self.metadata_buffer, col)
        return self.metadata_buffer

    def get_metadata_buffer(self):
        return self.metadata_buffer


    def get_steps(self):
        if len(self.dataframe)>0 :
            return self.cols_caracs[['step_id', 'step_label']].drop_duplicates()
        else:
            return 'empty df !'


    def sorted_columns(self, reverse = False):
        sorted_cols = sorted(list(self.dataframe.columns), reverse=reverse)
        return sorted_cols

    def sorted_columns_from_keys(self, list_of_keys):
        sorted_cols = list_of_keys
        for col in self.dataframe.columns:
            if not col in sorted_cols:
                sorted_cols.append(col)
        return sorted_cols

    def convert_in_list(self, dataframe, col):
        from ast import literal_eval
        """
        This function goal is to convert a pandas column into a "list" datatype column
        IMPORTANT : The column values must match with the python lists pattern in order to be read and converted correctly.

        RESULT : The same column, with each value converted into an array : that's also possible to loop over the array values

        PARAMS :
        - dataframe : the entry DataFrame
        - col : String, the column to convert
        """
        dataframe[col] = dataframe[col].apply(literal_eval)
        return dataframe
