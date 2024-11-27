import streamlit as st

from vanna.remote import VannaDefault

@st.cache_resource(ttl=3600)
def train_vanna():
    vn = VannaDefault(api_key=st.secrets.get("VANNA_API_KEY"), model=st.secrets.get("VANNA_MODEL"))
    # vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
    vn.connect_to_snowflake(
        account=st.secrets.get("SNOWFLAKE_ACCOUNT"),
        username=st.secrets.get("SNOWFLAKE_USER"),
        password=st.secrets.get("SNOWFLAKE_PASSWORD"),
        database=st.secrets.get("SNOWFLAKE_DATABASE"),
        role=st.secrets.get("SNOWFLAKE_ROLE"),
        schema=st.secrets.get("SNOWFLAKE_SCHEMA"),
    )
    # df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where table_schema = 'TPCH_SF1'")
    # plan = vn.get_training_plan_generic(df_information_schema)
    # # vn.get_training_plan_snowflake
    # plan
    # vn.train(plan=plan)
    training_plan = vn.get_training_plan_snowflake(filter_databases=['SNOWFLAKE_SAMPLE_DATA'], filter_schemas=['TPCH_SF1'])
    training_plan
    vn.train(plan=training_plan)
    return vn    

@st.cache_resource(ttl=3600)
def setup_vanna():
    vn = VannaDefault(api_key=st.secrets.get("VANNA_API_KEY"), model=st.secrets.get("VANNA_MODEL"))
    # vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
    vn.connect_to_snowflake(
        account=st.secrets.get("SNOWFLAKE_ACCOUNT"),
        username=st.secrets.get("SNOWFLAKE_USER"),
        password=st.secrets.get("SNOWFLAKE_PASSWORD"),
        database=st.secrets.get("SNOWFLAKE_DATABASE"),
        role=st.secrets.get("SNOWFLAKE_ROLE"),
        schema=st.secrets.get("SNOWFLAKE_SCHEMA"),
    )
    # df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where table_schema = 'TPCH_SF1'")
    # plan = vn.get_training_plan_generic(df_information_schema)
    # # vn.get_training_plan_snowflake
    # plan
    # vn.train(plan=plan)
    # training_plan = vn.get_training_plan_snowflake(filter_databases=['SNOWFLAKE_SAMPLE_DATA'], filter_schemas=['TPCH_SF1'])
    # training_plan
    # vn.train(plan=training_plan)
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)