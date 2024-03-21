import dash as dash
from dash import dcc
from dash import html
import seaborn as sns
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import json

# Helper functions
def construct_options(option_list):
    # Given list of options, generate dict for options
    result = []
    for option in option_list:
        result.append({'label':option,'value':option})

    return result


def construct_options_business(option_list):
    result = []
    for index,option in option_list.iterrows():
        result.append({'label': option['name'], 'value': option['business_id']})

    return result


def hours_to_string(hours_dict):
    result = 'Opening Hours: \n \n'

    close_count = 0
    for key in hours_dict.keys():
        line = ''

        if hours_dict[key] == '0:0-0:0':
            line = f'{key}: Closed \n'
            close_count += 1
        else:
            line = f'{key}: {hours_dict[key]} \n'

        result += line

    if close_count > 6:
        result = 'Opening hours unavailable...'

    return result


def dates_to_counts(dates_df):
    dates_df = dates_df.apply(convert_datestring, axis=1)

    dates_df['temp_col'] = 1

    dates_df = dates_df.groupby(0).count()

    dates_df.columns = ['count']

    return dates_df


def shapiro_test(x):
    stats, p = shapiro(x)
    alpha = 0.01
    if p > alpha:
        return f"normal with Shapiro test result of {round(stats,2)} and p-value of {round(p,2)} with alpha {alpha}"
    else:
        return f"NOT normal with Shapiro test result of {round(stats,2)} and p-value of {round(p,2)} with alpha {alpha}"


def convert_datestring(row):
    row[0] = pd.to_datetime(row[0].replace('numpy.datetime64','').strip('\'()'))
    return row


# Plot colors
discrete_colors = ['#d32323', '#0073bb',  '#AB63FA', '#B6E880', '#027a97', '#fb433c', 'goldenrod', '#636EFA', '#EF553B', '#00CC96', '#FFA15A', '#19D3F3', '#FF6692', '#FF97FF', '#FECB52']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://necolas.github.io/normalize.css/8.0.1/normalize.css','/assets/style.css']

my_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
my_app.title = "Yelp Data Dashboard"

df = pd.read_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_review_fused.gzip',compression='gzip',index_col=[0])
df_category = pd.read_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_category_counts.gzip',compression='gzip',index_col=[0])

# Categorize if business is 'highly rated' (Above average)
df['highly_rated'] = df['stars'] >= df['stars'].mean()

# Main Layout
my_app.layout = html.Div([
    # Header section
    html.Header(children=[
        html.Div([
            html.Img(src='https://s3-media0.fl.yelpcdn.com/assets/public/logo_desktop_white.yji-2d2a7a4342fcfc9007c4020b1f76558a.svg', className='head_img'),
            html.H1('Data Dashboard'),
        ], className='head_left'),
        html.Div([
            html.H2('CS5763 - Information Visualization'),
            html.H3('Jude (Ken Yoong) Lim - lkyoong428@vt.edu'),
        ],className='head_right'),
    ]),

    # Intro content
    html.Div(children=[
        html.H4('Introduction'),
        html.P( 'The Yelp dataset is a subset of their businesses, reviews, and user data for use in personal, educational, and academic purposes. This dashboard visualizes data for the 52k restaurants contained in the data.', className='subheader')
    ], className='subhead-wrap'),

    # Tabbed interface
    dcc.Tabs(id='tab_number', children=[
        dcc.Tab(label='Star Distribution in States', value='tab1'),
        dcc.Tab(label='The Pie Rack', value='tab2'),
        dcc.Tab(label='Explore Categories', value='tab3'),
        dcc.Tab(label='Top 100 Restaurants', value='tab4'),
        dcc.Tab(label='Normality of Yelp', value='tab5'),
    ],value='tab1'),
    html.Div(id='tab-content'),

    html.Footer('Developed on Plotly + Dash')
])

state_dropdown = construct_options(df['state'].unique())

category_dropdown = construct_options(df_category.index.drop(['Restaurants','Food'])[:30])

business_dropdown = construct_options_business(df.sort_values(by='review_count', ascending=False)[:100])

t5_features_dropdown = ['stars','review_count','star_1','star_2','star_3','star_4','star_5','est_business_age','hours_per_week']
t5_features_dropdown = construct_options(t5_features_dropdown)

tab1 = html.Div([
    html.H4('Star Value Distributions'),
    html.P('Yelp  houses reviews for many businesses across all states. Here we explore the star rating distribution of restaurants located in the 26 states included in the public Yelp dataset along with their associated 4M reviews.',className='section-sub'),
    html.Div(children=[
        html.H5('Select up to 5 States'),
        html.P('Compare the star rating distribution of different states.'),
        # Histograms for star values
        dcc.Dropdown(id='state-drop-t1', options=state_dropdown, placeholder='Select State...',
                     multi=True, value=['PA']),
        html.H5('Filter Results'),
        html.Div([
            dcc.Checklist(['Highly Rated', 'Open only'], ['Open only'], inline=True, id='filter-t1'),
            html.Button('Toggle Stack/Group', id='stack-btn-t1', n_clicks=0),
        ], className='t1-controls'),
        dcc.Graph(id='figure-t1', mathjax=True)
    ]),
])

tab2 = html.Div([
    html.H4('A variety of pies'),
    html.P('How well rated are restaurants in a certain state? The following pie charts allow for the comparison of the proportion of highly rated restaurants in 2 states. Highly rated restaurants are defined as restaurants with a 3.5 or higher rating.',className='section-sub'),

    html.Div(children=[

        html.H5('Select 2 states'),
        html.P('Compare the proportion of highly rated to below average restaurants.'),
        dcc.Dropdown(id='state-drop-t2', options=state_dropdown, placeholder='Select State...',
                     multi=True, value=['PA', 'MO']),

        html.H5('Pick a range for business age'),
        # Age slider
        dcc.RangeSlider(0, 20, 1, value=[1, 5], id='age-slider-t2'),

        dcc.Graph(id='figure-t2', mathjax=True)
    ]),

])

tab3 = html.Div([
    html.H4('Explore restaurant categories'),
    html.P('There are over 700 categories of restaurants in the Yelp dataset. Here we dive deeper into the Top 30 categories, finding where they are located geographically as well as how they are distributed across the states.',className='section-sub'),

    html.Div(children=[
        html.H5('Discover more about the top 30 restaurant categories'),
        html.P('Select a category of restaurants to find out more about where they\'re located!'),

        dcc.Dropdown(id='category-drop-t3', options=category_dropdown, placeholder='Select category...', value='Nightlife'),

        html.P('Select a category to find out more', id='cat-detail-t3'),

        # Total count, map number in each state, score distribution
        dcc.Graph(id='figure-t3', mathjax=True),
        dcc.Graph(id='figure-2-t3', mathjax=True)
    ]),
])

tab4 = html.Div([
    html.H4('Most popular restaurants'),
    html.P('How do restaurants perform over-time individually? Here we picked the 100 restaurants with the most reviews and visualize their checkins and reviews over time.', className='section-sub'),

    html.Div(children=[
        html.H5('Select a restaurant'),
        html.P('Examine the reviews and checkin timelines for the most popular restaurants!'),

        # Business dropdowns
        dcc.Dropdown(id='business-drop-t4', options=business_dropdown, placeholder='Select Business...', value=business_dropdown[0]['value']),

        # Open/closed
        html.P('', id='open-t4'),

        html.Div([
            # Opening hours
            dcc.Textarea(disabled=True, id='hours-t4', placeholder='Please select a business'),

            # Review star distribution
            dcc.Graph(id='figure-t4', mathjax=True),
        ], className='horizontal-t4'),

        html.Div([
            html.Div([
                html.H5('Number of Bins for Histogram'),
                dcc.Input(id='numfield-t4', type='number', value='50'),
            ], className='t4-left'),
            html.Div([
                html.H5('Pick Date Range'),
                # Date picker
                dcc.RadioItems(['Before', 'Between', 'From', 'All Time'], 'Before', id='date-radio-t4', inline=True),

                dcc.DatePickerSingle(id='date-single-t4', date='2015-05-05', className='show'),

                dcc.DatePickerRange(id='date-range-t4', start_date='2015-05-05', end_date='2020-02-02',
                                    className='hide'),
            ], className='t4-mid'),
            html.Div([
                # Radio for review/checkin
                html.H5('Type of Data'),
                dcc.RadioItems(['Reviews', 'Checkins'], 'Reviews', id='business-radio-t4', inline=True),
            ], className='t4-right'),
        ], className='controls-t4'),


        # Review and checkin timeline for selected business, side by side subplot, histogram per day, cumsum line graph
        dcc.Graph(id='figure-2-t4', mathjax=True),
    ])
])

tab5 = html.Div([
    html.H4('Normality Check'),
    html.P("But how are these restaurants' scores and other properties distributed?",className='section-sub'),

    html.Div(children=[
        html.H5('Select feature to assess: '),
        dcc.Dropdown(id='feature-t5', options=t5_features_dropdown,
                     placeholder='Select feature...', value='stars'),

        html.H5('Select additional dimension: '),
        dcc.Dropdown(id='dimension-t5', options=[{'label': 'none', 'value': 'none'},
                                                 {'label': 'is_open', 'value': 'is_open'},
                                                 {'label': 'highly_rated', 'value': 'highly_rated'},
                                                 {'label': 'state', 'value': 'state'}],
                     placeholder='Select dimension...', value='none'),

        html.H5('What kind of plot would you like to see? Choose between Boxplot, Histogram and Violin'),
        dcc.Input(id='textfield-t5', type='text', value='Boxplot'),

        # Box, violin, histogram
        dcc.Graph(id='figure-t5', mathjax=True)
    ])
])

tabs = {
    'tab1': tab1,
    'tab2': tab2,
    'tab3': tab3,
    'tab4': tab4,
    'tab5': tab5
}

# Callbacks
@my_app.callback(
    Output(component_id='tab-content', component_property='children'),
    Input(component_id='tab_number', component_property='value')
)

def tab_swapper(input):
    return tabs[input]


# Dropdown handler
@my_app.callback(
    Output(component_id="state-drop-t1", component_property="options"),
    [
        Input(component_id="state-drop-t1", component_property="value"),
    ]
)
def update_dropdown_options(values):
    if len(values) > 4:
        retain_vals = values[:5]
        modified_drop = []

        for option in retain_vals:
            modified_drop.append({'label': option, 'value': option, 'disabled': False})

        return modified_drop
    else:
        return state_dropdown


# Generate state wise distribution graph
@my_app.callback(
    Output(component_id="figure-t1", component_property="figure"),
    [
        Input(component_id="state-drop-t1", component_property="value"),
        Input(component_id="filter-t1", component_property="value"),
        Input(component_id="stack-btn-t1", component_property="n_clicks"),
    ],
)
def t1_graph(states, filters, button_clicks):
    if type(states) == str:
        states = [states]

    if type(filters) == str:
        filters = [filters]

    temp_data = df[df['state'].isin(states)]

    filter_text = ' '

    for filter in filters:
        if filter == 'Open only':
            temp_data = temp_data[temp_data['is_open'] == 1]
            filter_text += 'Open '
        elif filter == 'Highly Rated':
            temp_data = temp_data[temp_data['highly_rated']]
            filter_text += 'Highly Rated '

    fig = go.Figure()

    if (button_clicks%2>0):
        fig = px.histogram(temp_data, x="stars", color="state", title=f'Star rating distribution for{filter_text}restaurants in {",".join(states)}', color_discrete_sequence=discrete_colors)

        fig.update_xaxes(title="State")
        fig.update_yaxes(title="Count")
        fig.update_layout(legend_title_text="State", title_font_size=20, title_x=0.5)

        return fig
    else:
        for index, state in enumerate(states):
            fig.add_trace(go.Histogram(x=temp_data[temp_data['state'] == state]['stars'], name=state, marker_color=discrete_colors[index]))

            fig.layout.title = f'Star rating distribution for{filter_text}restaurants in {",".join(states)}'

            fig.update_xaxes(title="State")
            fig.update_yaxes(title="Count")

            fig.update_layout(legend_title_text="State", title_font_size=20, title_x=0.5)

    return fig


# Dropdown handler for t2
@my_app.callback(
    Output(component_id="state-drop-t2", component_property="options"),
    [
        Input(component_id="state-drop-t2", component_property="value"),
    ]
)
def update_dropdown_options2(values):
    if len(values) > 1:
        retain_vals = values[:2]
        modified_drop = []

        for option in retain_vals:
            modified_drop.append({'label': option, 'value': option, 'disabled': False})

        return modified_drop
    else:
        return state_dropdown


# Make pies for t2
@my_app.callback(
    Output(component_id="figure-t2", component_property="figure"),
    [
        Input(component_id="state-drop-t2", component_property="value"),
        Input(component_id="age-slider-t2", component_property="value"),
    ],
)
def t2_graph(states, ages):
    if type(states) == str:
        states = [states]

    state_counts = df['state'].value_counts()

    pull_values = []

    for state in state_counts.index:
        if state in states:
            pull_values.append(0.2)
        else:
            pull_values.append(0)

    temp_data = df[df['state'].isin(states)][(df['est_business_age'] >= ages[0]) & (df['est_business_age'] <= ages[1])]

    filter_text = f'{ages[0]} and {ages[1]}'

    # Left business distribution in states, highly rated businesses in each state 1 row 3 cols

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]], horizontal_spacing=0.01)

    fig.add_trace(go.Pie(labels=state_counts.keys(), values=state_counts.values, pull=pull_values, title=f"Overall distribution of restaurants", name="Restaurants", marker_colors=discrete_colors), row=1, col=1)

    for index,state in enumerate(states):
        col_number = 2 + index
        pie_vals = temp_data[temp_data['state'] == state]['highly_rated'].value_counts()
        pie_vals['Highly Rated'] = pie_vals.pop(True)
        pie_vals['Below Avg.'] = pie_vals.pop(False)
        fig.add_trace(go.Pie(labels=pie_vals.keys(), values=pie_vals.values, title=f"Proportion of Highly Rated Restaurants in {state}", name=f"Restaurants", marker_colors=discrete_colors), row=1, col=col_number)

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title=f'Comparing restaurants in {" and ".join(states)} between {filter_text} years old', title_font_size=20,title_x=0.5)

    return fig


# Make scatterplot and bar chart for categories
@my_app.callback(
    [Output(component_id="cat-detail-t3", component_property="children"),
     Output(component_id="figure-t3", component_property="figure"),
     Output(component_id="figure-2-t3", component_property="figure"),
     ],
    [
        Input(component_id="category-drop-t3", component_property="value"),
    ],
)
def t3_graph(category):
    temp_data = df[df['categories'].str.contains(category, regex=False)]

    detail_string = f'Found {len(temp_data)} {category} restaurants'

    fig = px.scatter_geo(temp_data,
                       lat='latitude',
                       lon='longitude',
                       color='state',
                       hover_name="name",
                       height=600,
                        title=f'Locations of {category} restaurants',
                         color_discrete_sequence=discrete_colors)

    fig.update_layout(
        geo_scope='usa',
        geo_showland=True,
        title_font_size = 20,
        title_x = 0.5
    )

    fig.update_layout(legend_title_text="State")

    fig2 = px.histogram(temp_data, y="state", color="highly_rated", title=f'Number of {category} restaurants in each state', color_discrete_sequence=discrete_colors)

    fig2.update_xaxes(title="Count")
    fig2.update_yaxes(title="State")
    fig2.update_layout(legend_title_text="Highly Rated",title_font_size=20,title_x=0.5)

    return detail_string, fig, fig2


# Update text areas, divs, make graphs
@my_app.callback(
    [Output(component_id="open-t4", component_property="children"),
     Output(component_id="hours-t4", component_property="value"),
     Output(component_id="figure-t4", component_property="figure"),
     Output(component_id="figure-2-t4", component_property="figure"),
     Output(component_id="date-single-t4", component_property="className"),
     Output(component_id="date-range-t4", component_property="className")
     ],
    [
        Input(component_id="business-drop-t4", component_property="value"),
        Input(component_id="business-radio-t4", component_property="value"),
        Input(component_id="date-radio-t4", component_property="value"),
        Input(component_id="date-single-t4", component_property="date"),
        Input(component_id="date-range-t4", component_property="start_date"),
        Input(component_id="date-range-t4", component_property="end_date"),
        Input(component_id="numfield-t4", component_property='value')
    ],
)
def t4_graph(business, data_type, date_cat, date_single, date_start, date_end, nbins):
    dtype = ''
    if data_type == 'Reviews':
        dtype = 'review_dates'
        date_data = pd.DataFrame(df[df['business_id'] == business][dtype].values[0].strip('][').split(', '))
    elif data_type == 'Checkins':
        dtype = 'checkins'
        date_data = pd.DataFrame(df[df['business_id'] == business][dtype].values[0].strip('][').split(', '))

    bname = df[df['business_id'] == business]['name'].values[0]
    bstate = df[df['business_id'] == business]['state'].values[0]
    bcity = df[df['business_id'] == business]['city'].values[0]
    bopen = df[df['business_id'] == business]['is_open'].values[0]
    bstars = df[df['business_id'] == business]['stars'].values[0]
    print(df[df['business_id'] == business]['hours'].to_dict())
    bhours = json.loads(list(df[df['business_id'] == business]['hours'].to_dict().values())[0].replace('\'','"'))
    bhours = hours_to_string(bhours)
    bstarcounts = df[df['business_id'] == business][['star_1', 'star_2', 'star_3', 'star_4', 'star_5']].T
    bstarcounts.columns = ['count']
    date_data = dates_to_counts(date_data)

    if bopen == 1:
        bopen = f"{bname} in {bcity}, {bstate}, rated {bstars} stars is open for business!"
    else:
        bopen = f"{bname} in {bcity}, {bstate}, rated {bstars} stars is permanently closed..."

    fig = px.histogram(x=['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'], y=bstarcounts['count'], title=f"Star rating distribution for {bname} in {bcity}, {bstate}", color_discrete_sequence=discrete_colors)
    fig.update_xaxes(title_text='Star Rating')
    fig.update_yaxes(title_text='Count')
    fig.update_layout(
        title_font_size = 20,
        title_x = 0.5
    )

    single_class = ''
    range_class = ''
    date_string = ''

    # Filter dates
    if date_cat == 'Before':
        single_class = 'show'
        range_class = 'hide'
        date_string = f'before {date_single}'
        date_data = date_data[date_data.index < pd.to_datetime(date_single)]
    elif date_cat == 'Between':
        single_class = 'hide'
        range_class = 'show'
        date_string = f'between {date_start} & {date_end}'
        date_data = date_data[(date_data.index > pd.to_datetime(date_start)) & (date_data.index < pd.to_datetime(date_end))]
    elif date_cat == 'From':
        single_class = 'show'
        range_class = 'hide'
        date_string = f'from {date_single}'
        date_data = date_data[date_data.index > pd.to_datetime(date_single)]
    elif date_cat == 'All Time':
        single_class = 'hide'
        range_class = 'hide'
        date_string = f'from all time'

    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.05, subplot_titles=(f"{data_type} over time for {bname} in {bcity}, {bstate}", f"Cumulative {data_type} over time for {bname} in {bcity}, {bstate}"))
    fig2.add_trace(go.Histogram(x=date_data.index, y=date_data['count'], name=f'{data_type} in a day', nbinsx=int(nbins), marker_color=discrete_colors[0]), row=1, col=1)
    fig2.add_trace(go.Line(x=date_data.index, y=date_data.cumsum()['count'], name=f'Cumulative sum of {data_type}', line_color=discrete_colors[0]), row=1, col=2)
    fig2.update_layout(showlegend=False, title=f'{data_type} for {bname} in {bcity}, {bstate} {date_string}', title_font_size = 20, title_x = 0.5)

    return bopen, bhours, fig, fig2, single_class, range_class


# Update text areas, divs, make graphs
@my_app.callback(
     Output(component_id="figure-t5", component_property="figure"),
    [
        Input(component_id="feature-t5", component_property="value"),
        Input(component_id="dimension-t5", component_property="value"),
        Input(component_id="textfield-t5", component_property="value"),
    ],
)
def t5_graph(feature, dimension, plottype):
    ptype = plottype.lower()

    if dimension != 'none':
        if ptype == 'histogram':
            fig = px.histogram(df, x=feature, color=dimension, color_discrete_sequence=discrete_colors)

        elif ptype == 'boxplot':
            fig = px.box(df, x=feature, color=dimension, color_discrete_sequence=discrete_colors)

        elif 'violin' in ptype:
            fig = px.violin(df, x=feature, color=dimension, color_discrete_sequence=discrete_colors)

        fig.update_layout(title=f'Distribution of {feature}: {shapiro_test(df[feature])}, split on {dimension}', title_font_size=20, title_x=0.5)
        return fig
    else:
        if ptype == 'histogram':
            fig = px.histogram(df, x=feature, color_discrete_sequence=discrete_colors)

        elif ptype == 'boxplot':
            fig = px.box(df, x=feature, color_discrete_sequence=discrete_colors)

        elif 'violin' in ptype:
            fig = px.violin(df, x=feature, color_discrete_sequence=discrete_colors)

        fig.update_layout(title=f'Distribution of {feature}: {shapiro_test(df[feature])}', title_font_size=20, title_x=0.5)
        return fig


my_app.server.run(port=8021, host='localhost')




