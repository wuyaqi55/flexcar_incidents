import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_selector as selector
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
import snowflake.connector
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

# data.info(verbose=True)
## help functions
def get_attribute_from_snowflake():
    conn = snowflake.connector.connect(
        user='YAQIWU11235',
        password='nut3yub2tzh7MPA.dnc',
        account='MCB49099-KHB29787',
        warehouse='OSCILAR_TO_SNOWFLAKE',
        database = 'FLEXCAR_OSCILAR',
        schema = 'OSCILAR_SCHEMA'
        )

    accident_df = pd.read_csv('../data/incident_data_all.csv')


    columns = ["Subscriber Id",
            "Created At",
                "ClearID_TotalScore",
                "RiskInform_TotalScore",
                "RiskInform_AddressFlags_TotalScore",
                "RiskInform_CriminalFlags_TotalScore",
                "RiskInform_Custom_TotalScore",
                "RiskInform_SingleRiskIndicators_TotalScore",
                "RiskInform_SyntheticIdentity_TotalScore"
            ]

    q = f"""
        WITH tr AS (
                SELECT
                    "Subscriber Id",
                    "Created At",
                    "ClearID_TotalScore",
                    "RiskInform_TotalScore",
                    "RiskInform_AddressFlags_TotalScore",
                    "RiskInform_CriminalFlags_TotalScore",
                    "RiskInform_Custom_TotalScore",
                    "RiskInform_SingleRiskIndicators_TotalScore",
                    "RiskInform_SyntheticIdentity_TotalScore",
                FROM
                    tr_attributes
                WHERE
                    "Subscriber Id" in {tuple(accident_df['Subscriber Id'])}
                ),
                
        combine_data as (
            select tr."Subscriber Id" as subscriberId, 
                tr."Created At" as createdAt,
                * 
            from tr
            join METHOD_ATTRIBUTES as m 
                on tr."Subscriber Id" = m."Subscriber Id" 
                and tr."Created At" = m."Created At"
            join SENTILINK_ATTRIBUTES as s 
                on tr."Subscriber Id" = s."Subscriber Id"
                and tr."Created At" = s."Created At"
        ),
        combine_data_rn as (
            select *,  
                row_number() over(partition by subscriberId order by createdAt) as rn
            from combine_data
        )
    select * 
    from combine_data_rn 
    where rn = 1;
    """

    # get the column names
    # q_col = "show columns in tr_attributes;"

    with conn.cursor() as cur:
        df = cur.execute(q).fetchall()
        # col_data = cur.execute(q_col).fetchall()

    # col_names = []
    # for col in col_data:
    #     col_names.append(col[2])
    # col_names.append('rn')

    # col_names = columns + ['rk']

    col_names = ["SUBSCRIBERID", "CREATEDAT", "Subscriber Id", "Created At", "ClearID_TotalScore", "RiskInform_TotalScore", 
                "RiskInform_AddressFlags_TotalScore", "RiskInform_CriminalFlags_TotalScore", "RiskInform_Custom_TotalScore", 
                "RiskInform_SingleRiskIndicators_TotalScore", "RiskInform_SyntheticIdentity_TotalScore", "Subscriber Id", "Created At", 
                "Subscriber_RunDate_JoinKey", "creditScore", "Flexscore", "PriorAuto_AtLeastOneGoodStanding", "PriorAuto_AtLeastOneMajorDelinquency", 
                "PriorAuto_AtLeastOnePastDue", "PriorAuto_AllInGoodStanding", "PriorAuto_Repossession", "PriorAuto_Chargeoff", "PriorAuto_DelinquentEvent", 
                "PriorCC_Chargeoff", "PriorCC_AtLeastOneGoodStanding", "PriorCC_AtLeastOneMajorDelinquency", "PriorCC_AtLeastOnePastDue", 
                "PriorCC_AllInGoodStanding", "PriorCC_Chargeoff_or_PriorAuto_Repo", "PriorCC_Chargeoff_or_PriorAuto_Chargeoff", "AnyPriorDelinquentEvent", 
                "AutoBalances", "AutoDebtLoad", "AutoMinimumPayment", "AutoOriginalLoanAmts", "CCBalances", "CCDebtLoad", "CCLimits", "CCMinimumPayment", 
                "MortgageBalances", "MortgageDebtLoad", "MortgageMinimumPayment", "MortgageOriginalLoanAmts", "StudentBalances", "StudentDebtLoad", 
                "StudentMinimumPayment", "StudentOriginalLoanAmts", "TotalDebtLoad", "TotalMinimumPayment", "MonthlyPaymentToIncome", "Subscriber Id", 
                "Created At", "Subscriber_RunDate_JoinKey", "creditScore", "Serial_Disputer", "Auth_User_Purchaser", "Flexscore", "FirstPartyFraud", "IDTheft", "RN"
    ]

    oscilar_data = pd.DataFrame(df, columns=col_names)
    return oscilar_data

# oscilar_data = pd.DataFrame(df)

def month_to_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'
    

def score_to_bascket(score):
    if score < 0.6:
        return 1
    elif 0.6 < score < 0.8:
        return 2
    else:
        return 3


def run_as_select(target = 'seasoning_3month_incident_Ind', data):
    """ 
    target need to be any of below:
        'seasoning_3month_mp_Ind', 
        'seasoning_month_mp_Ind', 
        'seasoning_3month_incident_Ind',
        'seasoning_month_incident_Ind',
    """
    data['vehivel_request_fufillment'] = data['Vehicle Requests Make Model List'] == data['Most Recent Vehicle Request Assigned Make Model']
    data['season'] = data['Month of Creation Date'].map(month_to_season)
    # data['seasoning_month_incident_Ind'] = data['days_run_in_accident'].map(lambda x: 1 if 0 < x < 30 else 0)
    data['seasoning_month_mp_Ind'] = data['days_run_in_mp'].map(lambda x: 1 if 0 < x < 30 else 0)
    data['score_basket'] = data['score_mean'].map(score_to_bascket)
    data['score_q1_vs_mean'] = data['score_q1'] - data['score_mean']
    data['score_q3_vs_mean'] = data['score_q3'] - data['score_mean']

    # columns to drop
    columns_to_drop = ['Most Recent Invoice Order Id', 'First Invoice Id', 'Most Recent Invoice Id', 'First Invoice Order Id', 'Subui Creation Date', ## id
                    '1st_incident', '1st_mp_date_customer_level', '1st_sp_date_customer_level', 'mp_counts_customer_level',  'sp_counts_customer_level',  # target relatd
                    'Creation Date', 'First Invoice Date', 'Most Recent Invoice Date', 'Driving Record Updated At', 'Driving Record Creation Time', 'Day of 1st_incident', # date and time
                    'Vehicle Requests Make Model List', 'Most Recent Vehicle Request Requested Make Model', 'Most Recent Vehicle Request Assigned Make Model',  'First Order Requested Make', # vehicle
                    'days_run_in_accident',  'Days With Vehicle', 'Days Since First Order', 'Active Weeks', 'days_run_in_mp', # days related to miles
                    'Total Tax Amount', 'Total Revenue', 'Total Collected Revenue', 'Total Collected Vehicle Amount', 'Total Receipts', 'Total Paid Invoices', 'Total Collected Mileage Amount', 'Total Collected Fee Amount', 'Total Paid Mileage Invoices', # revenue realted features
                        'Credit Score Approval Status', 
                    'License State',
                    'Most Recent Promo Code Used', # others
                    'Total Miles Driven Monthly', 'Total Miles Driven Pre Monthly', 'Miles per Week', 'Total Miles Driven', # miles
                    # 'seasoning_3month_mp_Ind', 
                    'seasoning_month_mp_Ind', 
                    'seasoning_3month_incident_Ind',
                    'seasoning_month_incident_Ind',
                    'Miles per Month', 'Miles Driven', # miles driven
                    'score_q1', 'score_q2', 'score_q3', 
                    # 'score_mean',
                    ]

    subscriber_col = []
    for col in data.columns:
        if 'subscriber' in col.lower():
            subscriber_col.append(col)

    # Target = 'seasoning_month_mp_Ind'
    Target = target

    columns_to_drop.remove(Target)

    y = data[Target]
    X = data.drop(columns = [Target] + subscriber_col +columns_to_drop, axis=1)

    temp = data.drop(columns = subscriber_col +columns_to_drop, axis=1)
    temp.to_csv('../data/h2o_incident_data.csv', index=False)
    temp1 = temp[temp['seasoning_3month_incident_Ind'] == 1]
    temp1.to_csv('../data/h2o_mp_data.csv', index=False)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    miles_columns = []

    numerical_columns_selector = selector(dtype_include=np.number)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(X_train)
    categorical_columns = categorical_columns_selector(X_train)

    # update num and cat columns
    numerical_columns = list(set(numerical_columns) - set(miles_columns))
    categorical_columns = list(set(categorical_columns) - set(miles_columns))

    # preprocesser
    numeric_transformer = Pipeline(
        steps=[("imputer_num", 
                SimpleImputer(strategy="median")), 
                # SimpleImputer(strategy="constant", fill_value=0)), 
                ("scaler", StandardScaler())]
    )

    miles_transformer = Pipeline(
        steps=[("imputer_num", 
                SimpleImputer(strategy="constant", fill_value=0)), 
                ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("imputer_cat", SimpleImputer(strategy="constant", fill_value='missing')), 
            ("encode", OneHotEncoder(handle_unknown='ignore',drop='if_binary'))]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numerical_columns)),
            ("miles", miles_transformer, list(miles_columns)),
            ("cat", categorical_transformer, list(categorical_columns)),
        ]
    )

    steps = [('preprocess', preprocessor),
            #  ('oversample', RandomOverSampler(random_state=42)),
            # ('rf', XGBClassifier())
            ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=20, min_samples_leaf=1, min_samples_split=2))
            ]

    model = Pipeline(steps)
    model_rf = model.fit(X_train, y_train)

    ## feature impartant
    # feature_importances = pd.DataFrame(model_rf[1].feature_importances_, index = model_rf[:-1].get_feature_names_out())
    # feature_importances.sort_values(by=0, ascending=False).head(20)
    feature_importances = pd.DataFrame(model_rf[1].feature_importances_, index = model_rf[:-1].get_feature_names_out())
    print('Feature impartance:')
    print('---------------------------------------------------------')
    print(feature_importances.sort_values(by=0, ascending=False).head(20))
    print('\n')

    ## confusion matrix
    from sklearn.metrics import roc_auc_score
    print('---------- ROC_AUC_score-------------')
    print('Training data:', roc_auc_score(y_train, model_rf.predict_proba(X_train)[:, 1]))
    print('Test data:', roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1]))
    print('\n')

    from sklearn.metrics import matthews_corrcoef
    print('---------- MCC_score-------------')
    print('Training data:', matthews_corrcoef(y_train, model_rf.predict(X_train)))
    print('Test data:', matthews_corrcoef(y_test, model_rf.predict(X_test)))
    print('\n')

    from sklearn.metrics import confusion_matrix
    print('------------------- Confusion Matric -------------------')
    y_test_pred= model_rf.predict(X_test)
    # print(confusion_matrix(y_test, y_test_pred, labels=[0,1]))
    y_test.name = 'Actual'
    y_test_pred = pd.Series(y_test_pred, name='Predict')
    print(pd.crosstab(y_test, y_test_pred))
    print('\n')


    from sklearn.metrics import classification_report
    print('------------------- classification_report -------------------')
    target_names = ['0', '1']
    y_test_pred= model_rf.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    print('\n')


    ## shap
    import shap
    X_transformed = model_rf.named_steps['preprocess'].transform(X_train)
    clf = model_rf.named_steps["rf"]
    columns = model_rf.named_steps['preprocess'].get_feature_names_out()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    shap.summary_plot(shap_values[:,:,1], X_transformed, feature_names=columns)
    # X = data.drop(columns=[Target]+['Subscriber Id'], axis=1)


if __name__ == '__main__':
    data_name = 'incident_train_data/incident_data_all_10.6_train.csv'
    path = '../data/' + data_name
    data = pd.read_csv(path)
    # data.info(verbose=True)
    # dropping the missing data > 0.3
    missing_ratio = oscilar_data.isnull().mean() 
    col_to_drop_m = missing_ratio[missing_ratio>0.3].index.to_list()
    oscilar_data.drop(columns=col_to_drop_m, inplace=True)
    # remove unwanted columns
    oscilar_data = oscilar_data.iloc[:, :-1]


