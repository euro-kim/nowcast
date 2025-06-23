class Benchmarks:
    def __init__(self, 
                 rmse : float,
                 mae : float, 
                 r2 : float,
                 mape : float,
                 smape : float,
                 diff_log_rmse : float = None,
                 diff_log_mae : float = None, 
                 diff_log_r2 : float = None,
                 diff_log_mape : float = None,
                 diff_log_smape : float = None,
                 target_variable : str = None,
                 ):
        self.rmse = rmse
        self.mae = mae
        self.r2 = r2
        self.mape = mape
        self.smape = smape
        self.diff_log_rmse = diff_log_rmse
        self.diff_log_mae = diff_log_mae
        self.diff_log_r2 = diff_log_r2
        self.diff_log_mape = diff_log_mape
        self.diff_log_smape = diff_log_smape
        self.target_variable = target_variable


    def __repr__(self):
        text : str = f'\n------Predicting {self.target_variable}------\n'
        if self.diff_log_rmse:
            text += f"\nDiff Log RMSE:  {self.diff_log_rmse:.4f}"
            text += f"\nDiff Log MAE:   {self.diff_log_mae:.4f}"
            text += f"\nDiff Log R²:    {self.diff_log_r2:.4f}"
            text += f"\nDiff Log MAPE:  {self.diff_log_mape:.4f}%"
            text += f"\nDiff Log sMAPE: {self.diff_log_smape:.4f}%"
        if self.rmse:
            text += f"\nRestored RMSE:  {self.rmse:.4f}"
            text += f"\nRestored MAE:   {self.mae:.4f}"
            text += f"\nRestored R²:    {self.r2:.4f}"
            text += f"\nRestored MAPE:  {self.mape:.4f}%"
            text += f"\nRestored sMAPE: {self.smape:.4f}%"
        return text
    
    def to_dict(self):
        dic = {}
        if self.diff_log_rmse:
            dic['diff_log_rmse'] = self.diff_log_rmse
            dic['diff_log_mae'] = self.diff_log_mae
            dic['diff_log_r2'] = self.diff_log_r2
            dic['diff_log_mape'] = self.diff_log_mape
            dic['diff_log_smape'] = self.diff_log_smape
            dic['restored_rmse'] = self.rmse
            dic['restored_mae'] = self.mae
            dic['restored_r2'] = self.r2
            dic['restored_mape'] = self.mape
            dic['restored_smape'] = self.smape
        return dic
