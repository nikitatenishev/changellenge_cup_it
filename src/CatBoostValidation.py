import numpy as np
import pandas
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import StratifiedGroupKFold


class CatBoostValidation():
    """
    Provides CatBoostRanker validation
    """
    def __init__(
            self,
            x: pandas.core.frame.DataFrame,
            y: pandas.core.series.Series,
            all_groups: pandas.core.series.Series
    ) -> None:
        """
        Initializes class attributes
        :x: train data
        :y: target data
        :all_group: data groups
        :returns: None
        """
        self.__catboost = CatBoostRanker(
            iterations=100,
            custom_metric=["NDCG:top=5"],
            verbose=True,
            random_seed=33,
            loss_function="YetiRank",
            task_type="GPU",
            metric_period=20,
        )
        self.__kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)
        self.__x = x
        self.__y = y
        self.__all_groups = all_groups
        self.__scores = []

    def validate(self) -> None:
        """
        Validates model on 5 folds
        :returns: None
        """
        for train_index, test_index in self.__kfold.split(
            self.__x,
            self.__y,
            self.__all_groups
        ):

            train = Pool(
                data=self.__x.loc[train_index],
                label=self.__y[train_index],
                group_id=self.__all_groups[train_index].values
            )

            test = Pool(
                data=self.__x.loc[test_index],
                label=self.__y[test_index],
                group_id=self.__all_groups[test_index].values
            )

            self.__catboost.fit(train, eval_set=test, plot=False)

            self.__scores.append(
                self.__catboost.evals_result_["validation"]["NDCG:top=5;type=Base"][-1]
            )

            print(70 * '=')
        print(f"[mean NDCG@5]: {round(np.mean(self.__scores), 7)}")
