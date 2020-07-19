import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Logger(SummaryWriter):
    def __init__(self, trade_id, model_info=None, trade_info=None):
        super().__init__()
        self.trade_id = trade_id
        self.model_info = model_info
        self.trade_info = trade_info
        self.trade_history = []

    def append_trade_data(
        self,
        trade_num,
        trade_type,
        real_val,
        forecast_val,
        diff,
        lost_val,
        cash_at_hand,
    ) -> None:
        """
            Add one trade log to the history.
            Note that this writes each and every trade made by the trader hourly.
            This will be called every time one transaction occurs for a timestamp.
        """
        assert trade_type != None
        assert diff != None
        assert lost_val != None
        assert type(real_val) == float or type(real_val) == int
        assert type(forecast_val) == float or type(forecast_val) == int

        trade_data = {
            "trade_num": trade_num,
            "trade_info": {
                "trade_type": trade_type,
                "energy_info": {
                    "real_val_kW": real_val,
                    "forecast_val_kW": forecast_val,
                    "diff_in_energy": diff,
                },
                "loss": lost_val,
            },
            "current_cash_at_hand": cash_at_hand,
        }
        self.trade_history.append(trade_data)
        self.add_scalar("cash_at_hand", cash_at_hand, trade_num)

    def log_trade_history(self) -> None:
        """
            Log trade history into json format.
            This function does not contain any information about the model.
        """
        out_file_name = "src/trade/" + str(self.trade_id) + ".json"
        out_file_name = out_file_name.replace(":", "_")

        with open(out_file_name, "w") as outfile:
            json.dump(self.trade_history, outfile, indent=4, sort_keys=False)



    def draw_validation_result(self, val_y, pred_y, epoch):

        fig = plt.figure(1, figsize=(15, 7))
        plt.plot(val_y, color="blue")
        plt.plot(pred_y, color="red")
        plt.xlabel("time T", fontsize=20)
        plt.ylabel("Energy", fontsize=20)
        red_patch = mpatches.Patch(color="red", label="prediction")
        blue_patch = mpatches.Patch(color="blue", label="Actual")
        plt.legend(handles=[red_patch, blue_patch])

        self.add_figure("Validation/Pred", fig, epoch)

                                        
    def draw_lagged_correlation(self, vals):
        fig = plt.figure(3, figsize=(15,10))
        plt.plot(vals)
        plt.xlabel('Delta',fontsize=15)
        plt.ylabel('Correlation',fontsize=15)
        plt.title('Lag Correlation - Our Model',fontsize=15)
        self.add_figure("LaggedCorrelation", fig, 0)