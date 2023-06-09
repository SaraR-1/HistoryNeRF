fig = alt.Chart(data).mark_circle().encode(x='Number of Images', y='Rot. Error < 15°', tooltip=['Number of Images', 'Rot. Error < 15°', 'Rot. Error Mean', 'Rot. Error Median'])
st.altair_chart(fig, use_container_width=True)

fig = alt.Chart(data).mark_circle().encode(x='Number of Images', y='Transl. Error < 20%', tooltip=['Number of Images', 'Transl. Error < 20%', 'Transl. Error Mean', 'Transl. Error Median'])
st.altair_chart(fig, use_container_width=True)


x = [random() for x in range(500)]
y = [random() for y in range(500)]

s1 = ColumnDataSource(data=dict(x=x, y=y))
p1 = figure(width=400, height=400, tools="lasso_select", title="Select Here")
p1.circle('x', 'y', source=s1, alpha=0.6)

s2 = ColumnDataSource(data=dict(x=[], y=[]))
p2 = figure(width=400, height=400, x_range=(0, 1), y_range=(0, 1),
            tools="", title="Watch Here")
p2.circle('x', 'y', source=s2, alpha=0.6)

s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var d2 = s2.data;
        d2['x'] = []
        d2['y'] = []
        for (var i = 0; i < inds.length; i++) {
            d2['x'].push(d1['x'][inds[i]])
            d2['y'].push(d1['y'][inds[i]])
        }
        s2.change.emit();
    """)
)

layout = row(p1, p2)

st.bokeh_chart(layout)

    

source = ColumnDataSource(dict(
    mean_performance=[0.8, 0.7, 0.6, 0.5],
    median_performance=[0.8, 0.7, 0.6, 0.5],
    n_samples=[100, 200, 300, 400],
    experiment_number=[1, 2, 3, 4],
))

selected_idx = ColumnDataSource(
    dict(experiment_number=[1])
)

fig = figure(title='My Coordinates: Select the Circles',
             plot_height=500, plot_width=600,
             x_range=(0, 4), y_range=(0, 500))

plot = fig.circle(source=source, x='mean_performance', y='n_samples',
           color='green', size=10, alpha=0.5,
           hover_fill_color='black', hover_alpha=0.5)

tooltips = [('Mean Performance', '@mean_performance'), ('Samples', '@n_samples'), ('Median Performance', '@median_performance')]

fig.add_tools(HoverTool(tooltips=tooltips, renderers=[plot]))
fig.add_tools(TapTool())

callback = CustomJS( code="""
            document.dispatchEvent(
                new CustomEvent("POINT_SELECTED", {data: 'test'})
            )
        """
    )

source.js_on_event(Tap, callback)

event_result = streamlit_bokeh_events(fig, key="foo", events="POINT_SELECTED", refresh_on_update=True, debounce_time=0)

st.write(event_result)


# st.bokeh_chart(fig, use_container_width=True)
st.write('Selected Experiment Number: ', selected_idx.data)