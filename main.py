import streamlit as st
import pandas as pd
import numpy as np
from pygam import ExpectileGAM
import io
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib import pyplot, transforms
#pip install mpld3
import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
import pylab
import plotly.express as px
import plotly.io as pio
import cufflinks as cf
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.formula.api as smf
from st_aggrid import AgGrid
import streamlit.components.v1 as components
from pivottablejs import pivot_ui
from streamlit_option_menu import option_menu
from apps import home, heatmap, upload  # import your app modules here

st.set_page_config(layout="wide")


st.set_option('deprecation.showPyplotGlobalUse', False)

#Keep for trialing
ALL = 'ALL'
def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique


def main():
        """Semi Automated ML App with Streamlit """

        st.title("Geotechnical Data Analysis")
        st.text("Using Streamlit == 1.11.0")
        activities = ["EDA", "Plots", "Design Lines", "Geemaps", "About"]
        choice = st.sidebar.selectbox("Select Activities", activities)


        if choice == 'EDA':
            st.subheader("Exploratory Data Analysis")

            st.sidebar.subheader("Table options")
            height = st.sidebar.slider('Table Height', min_value=100, max_value=800, value=200)
            reload_data = False

            uploaded_file = st.file_uploader("Upload CSV with Location, Units and Depth as headers for full functionality", type=".csv")
            use_example_file = st.checkbox(
                "Use example file", False, help="Use in-built example file to demo the app")
            ab_default = None
            result_default = None

            # If CSV is not uploaded and checkbox is filled, use values from the example file
            # and pass them down to the next if block
            if use_example_file:
                uploaded_file = "iris_modified.csv"
                ab_default = ["variant"]
                result_default = ["converted"]
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                missing_values = ['??', 'na', 'N/A']  # add other repitive values if needed
                df = df.replace(missing_values, np.NaN)
                if st.checkbox("Show Table"):
                    ag = AgGrid(
                        df,
                        height=height,
                        fit_columns_on_grid_load=False
                    )

                if st.checkbox("Show Pivot"):
                    t = pivot_ui(df)

                    with open(t.src) as t:
                        components.html(t.read(), width=900, height=1000, scrolling=True)

                if st.checkbox("Summary"):
                    st.write("Units Only")
                    st.write(df.describe())
                    st.write("Testing All button")
                    makes = unique_sorted_values_plus_ALL(df.Units)
                    make_choice = st.selectbox('Select your Unit:', makes)
                    years = df["Location"].loc[df["Units"] == make_choice].unique()
                    year_choice = st.selectbox('(Optional) Select Location:', years)
                    depth = unique_sorted_values_plus_ALL(df.Depth)
                    depth_choice = st.sidebar.select_slider('Summary: (Optional) Select Depth:', depth)

                    df2 = df[(df["Units"] == make_choice)]
                    st.write(df2.describe())
                    df = df[
                        (df['Units'] == make_choice)]
                    df1 = df[
                        (df['Units'] == make_choice) & (df['Location'] == year_choice)]
                    df3 = df[
                        (df['Units'] == make_choice) & (df['Depth'] == depth_choice)]

                    st.write("Unit and Location")
                    st.write(df1.describe())
                    st.write("Unit and Depth")
                    st.write(df3.describe())
                if st.checkbox("Show Columns"):
                    all_columns = df.columns.to_list()
                    st.write(all_columns)

                if st.checkbox("Show Selected Columns"):
                    all_columns1 = df.columns.to_list()
                    selected_columns = st.multiselect("Select Columns", all_columns1)
                    new_df = df[selected_columns]
                    st.write("All Data")
                    st.dataframe(new_df)

        elif choice == 'Plots':
            st.subheader("Data Visualisation")
            uploaded_file = st.file_uploader("Upload CSV", type=".csv")
            use_example_file = st.checkbox(
                "Use example file", False, help="Use in-built example file to demo the app")
            ab_default = None
            result_default = None

            # If CSV is not uploaded and checkbox is filled, use values from the example file
            # and pass them down to the next if block
            if use_example_file:
                uploaded_file = "iris_modified.csv"
                ab_default = ["variant"]
                result_default = ["converted"]
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                col1, col2 = st.columns(2)
                x = list(df.select_dtypes(include=[np.number]).columns)[0:]
                y = list(df.select_dtypes(include=[np.number]).columns)[0:]
                category1 = list(df.select_dtypes('object').columns[0:])
                template = list(pio.templates.keys())
                themebox=list(cf.themes.THEMES.keys())
                colorscalebox=list(cf.colors._scales_names.keys())
                x_choice = st.sidebar.selectbox('Select your X-Axis:', x)
                y_choice = st.sidebar.selectbox('Select your Y-Axis:', y)
                category_choice = st.sidebar.selectbox('Select your Category:', category1)
                theme_choice = st.sidebar.selectbox('Select your Theme:', template)
                boxtheme_choice = st.sidebar.selectbox('Select your Box Theme:', themebox)
                boxcolour_choice = st.sidebar.selectbox('Select your Box Colour:', colorscalebox)
                fig = px.scatter(df, x=x_choice, y=y_choice, color=category_choice, opacity=0.5,
                                 render_mode='auto', template=theme_choice)
                # alternatively use box or violin for marginals
                # Below for cursor options/crosshairs

                fig.update_yaxes(autorange="reversed")
                # fig.update_traces(hovertemplate=None) #changes hoverdata to simple


                fig.update_layout(xaxis={'side': 'top'}, xaxis_title=x_choice.title(), yaxis_title=y_choice.title(), title=f'{y_choice.title()} vs {x_choice.title()}', title_x=0.0)

                col1.plotly_chart(fig)




                box1 = df.pivot(columns=category_choice, values=x_choice).figure(
                        kind='box', theme=boxtheme_choice, colorscale=boxcolour_choice,
                        yTitle=x_choice.title(),
                        xTitle = category_choice.title(),
                        title=f'{x_choice.title()} vs Unit')
                col2.plotly_chart(box1)

            # Customizable Plot

                type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"]

                if st.button("Generate Plot"):
                    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot, selected_columns_names))
                    if type_of_plot == 'area':
                        fig = px.area(df, x=x_choice, y=y_choice, color=category_choice)
                        st.plotly_chart(fig)

                    elif type_of_plot == 'bar':
                        fig1 = px.bar(df, x=x_choice, y=y_choice, color=category_choice)
                        st.plotly_chart(fig1)

                    elif type_of_plot == 'line':
                        fig2 = px.line(df, x=x_choice, y=y_choice, color=category_choice)
                        st.plotly_chart(fig2)
                    # Custom Plot
                    elif type_of_plot == 'scatter':
                        fig3 = px.scatter(df, x=x_choice, y=y_choice, color=category_choice, opacity=0.5,
                                         render_mode='auto', template=theme_choice)
                        # alternatively use box or violin for marginals
                        # Below for cursor options/crosshairs
                        st.plotly_chart(fig3)
                    elif type_of_plot == 'histogram':
                        fig4 = px.histogram(df, x=x_choice, color=category_choice, opacity=0.5,
                                            template=theme_choice)
                        # alternatively use box or violin for marginals
                        # Below for cursor options/crosshairs
                        st.plotly_chart(fig4)
                    elif type_of_plot == 'box':
                        fig5 = px.box(df, x=x_choice, y=y_choice, color=category_choice,
                                         template=theme_choice)
                        # alternatively use box or violin for marginals
                        # Below for cursor options/crosshairs
                        st.plotly_chart(fig5)
                    elif type_of_plot == 'kde':
                        fig5 = px.histogram(df, x=x_choice, y=y_choice, color=category_choice, opacity=0.5,
                                         marginal="box", template=theme_choice, hover_data=df.columns)
                        # alternatively use box or violin for marginals
                        # Below for cursor options/crosshairs
                        st.plotly_chart(fig5)

        elif choice == 'Design Lines':
            st.subheader("Design Lines")
            uploaded_file = st.file_uploader("Upload CSV", type=".csv")
            use_example_file = st.checkbox(
                "Use example file", False, help="Use in-built example file to demo the app")
            ab_default = None
            result_default = None

            # If CSV is not uploaded and checkbox is filled, use values from the example file
            # and pass them down to the next if block
            if use_example_file:
                uploaded_file = "iris_modified.csv"
                ab_default = ["variant"]
                result_default = ["converted"]
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                col1, col2 = st.columns(2)
                x = list(df.select_dtypes('number').columns)[0:]
                y = list(df.select_dtypes('number').columns)[0:]
                category1 = list(df.select_dtypes('object').columns[0:])
                x_choice = st.sidebar.selectbox('Select your X-Axis:', x)
                y_choice = st.sidebar.selectbox('Select your Y-Axis:', y)
                category_choice = st.sidebar.selectbox('Select your Category:', category1)
                fig = px.scatter(df, x=x_choice, y=y_choice, color=category_choice, opacity=0.5,
                                 render_mode='auto', template="plotly")
                # alternatively use box or violin for marginals
                # Below for cursor options/crosshairs
                fig.update_xaxes(showgrid=True, zeroline=True, rangeslider_visible=False, showticklabels=True,
                                 showspikes=True, spikemode='across', spikesnap='cursor', showline=True,
                                 spikecolor="grey", spikethickness=1, spikedash='solid')
                fig.update_yaxes(autorange="reversed", showspikes=True, spikedash='solid', spikemode='across',
                                 spikecolor="grey", spikesnap="cursor", spikethickness=1)
                # fig.update_traces(hovertemplate=None) #changes hoverdata to simple

                fig.update_layout(spikedistance=1000, hoverdistance=10)
                # below for drawing

                fig.update_layout(xaxis={'side': 'top'}, xaxis_title=x_choice.title(), yaxis_title=y_choice.title(), title=f'{y_choice.title()} vs {x_choice.title()}', title_x=0.0)
                fig.update_layout(
                    dragmode='drawopenpath',
                    newshape=dict(line_color='cyan'))
                config={'modeBarButtonsToAdd': ['drawline','drawopenpath','drawclosedpath',
                                                'drawcircle',
                                                'drawrect',
                                                'eraseshape'
                                                ]}
                col1.plotly_chart(fig, use_container_width=False, **{'config': config})

                figbox = px.box(df, y=y_choice, x=x_choice, color=category_choice, template="plotly")
                figbox.update_yaxes(autorange="reversed")
                figbox.update_traces(orientation='h')
                figbox.update_xaxes(showgrid=True, zeroline=True, rangeslider_visible=False, showticklabels=True,
                                  showspikes=True, spikemode='across', spikesnap='cursor', showline=True,
                                  spikecolor="grey", spikethickness=1, spikedash='solid')
                figbox.update_yaxes(showspikes=True, spikedash='solid', spikemode='across',
                                  spikecolor="grey", spikesnap="cursor", spikethickness=1)
                fig.update_layout(xaxis={'side': 'top'}, xaxis_title=x_choice.title(), yaxis_title=y_choice.title())
                # fig.update_traces(hovertemplate=None) #changes hoverdata to simple
                figbox.update_layout(spikedistance=1000, hoverdistance=500)
                # below for drawing
                figbox.update_layout(legend_traceorder="reversed",
                                   dragmode='drawopenpath',
                                   newshape=dict(line_color='cyan'),
                                   title_text='')
                config={'modeBarButtonsToAdd': ['drawline',
                                                          'drawopenpath',
                                                          'drawclosedpath',
                                                          'drawcircle',
                                                          'drawrect',
                                                          'eraseshape'
                                                          ]}
                col2.plotly_chart(figbox, use_container_width=False, **{'config': config})

                my_expander1 = st.expander(label='Loess')
                with my_expander1:

                    fig2 = px.scatter(df, x=y_choice, y=x_choice, color=category_choice, opacity=0.5,
                                     trendline="lowess", trendline_color_override="red", render_mode='auto')
                    st.plotly_chart(fig2, use_container_width=False, **{'config': config})
                    fig2.update_layout(
                        dragmode='drawopenpath',
                        newshape=dict(line_color='cyan'))
                    config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath',
                                                      'drawcircle',
                                                      'drawrect',
                                                      'eraseshape'
                                                      ]}

                my_expander2 = st.expander(label='Regression: Please select Unit first then Y and X')
                with my_expander2:

                    # We use a form to wait for the user to finish selecting columns.
                    # The user would press the submit button when done.
                    # This is done to optimize the streamlit application performance.
                    # As we know streamlit will re-run the code from top to bottom
                    # whenever the user changes the column selections.
                    with st.form('form'):
                        sel_column = st.multiselect('Select Unit, Y and X', df.columns,
                                                    help='Select a column to form a new dataframe. Press submit when done.')
                        sel_column1 = st.selectbox('Select Unit Type', df.Units.unique(),
                                                    help='Select a column to form a new dataframe. Press submit when done.')
                        drop_na = st.checkbox('Drop rows with missing value', value=True)
                        submitted = st.form_submit_button("Submit")

                    if submitted:
                        dfnew = df[sel_column]
                        if drop_na:
                            dfnew = dfnew.dropna()
                            t = sel_column1
                            dfnew = dfnew[dfnew.Units == t]
                        #st.write('New dataframe')
                        #dfnew_style = dfnew.style.format(precision=2, na_rep='MISSING', thousands=",",
                                                         #formatter={('A'): "{:.0f}"})
                        #st.dataframe(dfnew_style)

                        c1, c2 = st.columns(2)
                        c3, c4 = st.columns(2)

                        X = dfnew.iloc[:, 1]
                        y = dfnew.iloc[:, 2]

                        # lets fit the mean model first by CV
                        gam50 = ExpectileGAM(expectile=0.5)
                        gam50.gridsearch(X[:, None], y)

                        # and copy the smoothing to the other models
                        lam = gam50.lam

                        # now fit a few more models
                        gam95 = ExpectileGAM(expectile=0.95, lam=lam).fit(X, y)
                        #gam75 = ExpectileGAM(expectile=0.75, lam=lam).fit(X, y)
                        #gam25 = ExpectileGAM(expectile=0.25, lam=lam).fit(X, y)
                        gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)

                        XX = gam50.generate_X_grid(term=0, n=500)
                        fig = plt.figure(figsize=(5, 7))
                        ax = fig.add_subplot(1, 1, 1)
                        base1 = pyplot.gca().transData
                        rot1 = transforms.Affine2D().rotate_deg(-90)

                        with c1:
                            plt.scatter(X, y, c='k', alpha=0.5, transform=rot1 + base1)
                            plt.plot(XX, gam95.predict(XX), label='0.95', marker='.', transform=rot1 + base1)
                            plt.plot(XX, gam50.predict(XX), label='0.50', marker='.', transform=rot1 + base1)
                            plt.plot(XX, gam05.predict(XX), label='0.05', marker='.', transform=rot1 + base1)
                            ax.xaxis.set_ticks_position("top")
                            ax.xaxis.set_label_position('top')
                            ax.set_xlabel(x_choice.title())
                            ax.set_ylabel(y_choice.title())
                            ax.xaxis.tick_top()
                            ax.set_title(f'{y_choice.title()} vs {x_choice.title()}')
                            plt.legend()
                            buf = BytesIO()
                            fig.savefig(buf, format="png")
                            st.image(buf)
                            css = '''
                            table
                            {
                            border-collapse: collapse;
                            }
                            th
                            {
                            color: #ffffff;
                            background-color: #000000;
                            }
                            td
                            {
                            background-color: #cccccc;
                            }
                            table, th, td
                            {
                            font-family:Arial, Helvetica, sans-serif;
                            border: 1px solid black;
                            text-align: right;
                            }
                            '''

                        with c2:
                            for axes in fig.axes:
                                for line in axes.get_lines():
                                    xy_data = line.get_xydata()
                                    labels = []
                                    for x, y in xy_data:
                                        html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>x</th> <td>{x}</td> </tr> <tr> <th>y</th> <td>{y}</td> </tr> </tbody> </table>'
                                        labels.append(html_label)
                                    tooltip = plugins.PointHTMLTooltip(line, labels, css=css)
                                    plugins.connect(fig, tooltip)
                            fig_html = mpld3.fig_to_html(fig)
                            components.html(fig_html, height=850, width=850)
                            fn = 'scatter.png'
                            img = io.BytesIO()
                            plt.savefig(img, format='png')


                        x1 = dfnew.iloc[:, 1]
                        y1 = dfnew.iloc[:, 2]
                        smoothed = sm.nonparametric.lowess(exog=x1, endog=y1, frac=0.2)
                        fig5 = plt.figure(figsize=(5, 7))
                        ax = fig5.add_subplot(1, 1, 1)
                        base = pyplot.gca().transData
                        rot = transforms.Affine2D().rotate_deg(-90)
                        with c3:
                            plt.scatter(x1, y1, transform=rot + base)
                            ax.set_title(f'{y_choice.title()} vs {x_choice.title()}')
                            ax.set_xlabel(x_choice.title())
                            ax.set_ylabel(y_choice.title())
                            ax.xaxis.set_label_position('top')
                            ax.xaxis.tick_top()
                            plt.legend()
                            plt.plot(smoothed[:, 0], smoothed[:, 1], c="k", transform=rot + base)
                            buf = BytesIO()
                            fig5.savefig(buf, format="png")
                            st.image(buf)
                            css = '''
                            table
                            {
                            border-collapse: collapse;
                            }
                            th
                            {
                            color: #ffffff;
                            background-color: #000000;
                            }
                            td
                            {
                            background-color: #cccccc;
                            }
                            table, th, td
                            {
                            font-family:Arial, Helvetica, sans-serif;
                            border: 1px solid black;
                            text-align: right;
                            }
                            '''


                        new = dfnew.iloc[: , [1, 2]].copy()
                        new.columns = ['X', 'Y']
                        model = smf.quantreg('Y ~ X', new)
                        quantiles = [0.05, 0.95]
                        fits = [model.fit(q=q) for q in quantiles]
                        figure = plt.figure(figsize=(5, 7))
                        axes = figure.add_subplot(1, 1, 1)

                        x = new['X']
                        y = new['Y']

                        fit = np.polyfit(x, y, deg=1)
                        base = pyplot.gca().transData
                        rot = transforms.Affine2D().rotate_deg(-90)
                        smoothed = sm.nonparametric.lowess(exog=x, endog=y, frac=0.2)

                        _x = np.linspace(x.min(), x.max(), num=len(y))

                        # fit lines
                        _y_005 = fits[0].params['X'] * _x + fits[0].params['Intercept']
                        _y_095 = fits[1].params['X'] * _x + fits[1].params['Intercept']

                        # start and end coordinates of fit lines
                        p = np.column_stack((x, y))
                        a = np.array([_x[0], _y_005[0]])  # first point of 0.05 quantile fit line
                        b = np.array([_x[-1], _y_005[-1]])  # last point of 0.05 quantile fit line

                        a_ = np.array([_x[0], _y_095[0]])
                        b_ = np.array([_x[-1], _y_095[-1]])

                        # mask based on if coordinates are above 0.95 or below 0.05 quantile fitlines using cross product
                        mask = lambda p, a, b, a_, b_: (np.cross(p - a, b - a) > 0) | (np.cross(p - a_, b_ - a_) < 0)
                        mask = mask(p, a, b, a_, b_)

                        axes.scatter(x[mask], new['Y'][mask], facecolor='r', edgecolor='none', alpha=0.6,
                                     label='data point outside', transform=rot + base)
                        axes.scatter(x[~mask], new['Y'][~mask], facecolor='g', edgecolor='none', alpha=0.6,
                                     label='data point', transform=rot + base)

                        axes.plot(x, fit[0] * x + fit[1], label='best fit', c='lightgrey', marker='.', transform=rot + base)
                        axes.plot(_x, _y_095, label=quantiles[1], c='orange', marker='.', transform=rot + base)
                        axes.plot(_x, _y_005, label=quantiles[0], c='lightblue', marker='.', transform=rot + base)
                        axes.plot(smoothed[:, 0], smoothed[:, 1], label='lowess', c="k", marker='.', transform=rot + base)

                        axes.legend()
                        axes.xaxis.set_ticks_position('top')
                        axes.xaxis.set_label_position('top')
                        axes.set_xlabel(x_choice.title())
                        axes.set_ylabel(y_choice.title())
                        axes.set_title(f'{y_choice.title()} vs {x_choice.title()}')

                        buf = BytesIO()
                        figure.savefig(buf, format="png")
                        st.image(buf)
                        css = '''
                        table
                        {
                          border-collapse: collapse;
                        }
                        th
                        {
                          color: #ffffff;
                          background-color: #000000;
                        }
                        td
                        {
                          background-color: #cccccc;
                        }
                        table, th, td
                        {
                          font-family:Arial, Helvetica, sans-serif;
                          border: 1px solid black;
                          text-align: right;
                        }
                        '''
                        with c4:
                            for axes in figure.axes:
                                for line in axes.get_lines():
                                    xy_data = line.get_xydata()
                                    labels = []
                                    for x, y in xy_data:
                                        html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>x</th> <td>{x}</td> </tr> <tr> <th>y</th> <td>{y}</td> </tr> </tbody> </table>'
                                        labels.append(html_label)
                                    tooltip = plugins.PointHTMLTooltip(line, labels, css=css)
                                    plugins.connect(figure, tooltip)
                            fig_html = mpld3.fig_to_html(figure)
                            components.html(fig_html, height=850, width=850)
                            fn = 'scatter.png'
                            img = io.BytesIO()
                            plt.savefig(img, format='png')
        elif choice == 'Geemaps':
            st.subheader("Geemaps")
            apps = [{"func": home.app, "title": "Home", "icon": "house"},
                    {"func": heatmap.app, "title": "Heatmap", "icon": "map"},
                    {"func": upload.app, "title": "Upload", "icon": "cloud-upload"},
                   ]
            titles = [app["title"] for app in apps]
            titles_lower = [title.lower() for title in titles]
            icons = [app["icon"] for app in apps]

            params = st.experimental_get_query_params()

            if "page" in params:
                default_index = int(titles_lower.index(params["page"][0].lower()))
            else:
                default_index = 0

            with st.sidebar:
                selected = option_menu(
                    "Main Menu",
                    options=titles,
                    icons=icons,
                    menu_icon="cast",
                    default_index=default_index,
                )

                st.sidebar.title("About")
                st.sidebar.info(
                    """
                    This web [app](https://share.streamlit.io/giswqs/streamlit-template) is maintained by [Qiusheng Wu](https://wetlands.io). You can follow me on social media:
                        [GitHub](https://github.com/giswqs) | [Twitter](https://twitter.com/giswqs) | [YouTube](https://www.youtube.com/c/QiushengWu) | [LinkedIn](https://www.linkedin.com/in/qiushengwu).
        
                    Source code: <https://github.com/giswqs/streamlit-template>
                    More menu icons: <https://icons.getbootstrap.com>
                """
                )

            for app in apps:
                if app["title"] == selected:
                    app["func"]()
                    break                               
                                            
                    elif choice == 'About':
                        st.subheader("About")

if __name__ == '__main__':
    main()
