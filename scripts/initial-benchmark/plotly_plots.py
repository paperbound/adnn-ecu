import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import csv
c_perf = {
        'Network Size': [1600, 3190, 6370, 25450, 101770, 25818], 
        'Accuracy': [30.5, 56.82, 90.57, 94.85, 76.27, 94.67],
        'ExecTime': [1.39, 2.83, 6.06, 27.19, 141.74, 26.72], 
        'PeakMemory': [1079.2, 1079.28, 1079.42, 1080.32, 1083.92, 1080.35]
        }
c_df = pd.DataFrame(data = c_perf)
c_df['Implementation'] = 'C'

numpy_perf = {
            'Network Size': [1600, 6370, 25450, 101770], 
            'Accuracy': [35.94, 89.35, 95.32, 96.83], 
            'ExecTime': [32.85, 42.98, 95.73, 372.85], 
            'PeakMemory': [1301.1, 1301.1, 1301.1, 1301.1]
            }
numpy_df = pd.DataFrame(data = numpy_perf)
numpy_df['Implementation'] = 'Numpy'

eigen_perf = {
            'Network Size': [1600, 3190, 6370, 25450, 101770, 25818], 
            'Accuracy': [27.88, 54.71, 90.6, 95.24, 97.09, 95.51], 
            'ExecTime': [2.24, 4.09, 7.87, 31.0, 191.93, 31.77], 
            'PeakMemory': [752.83, 752.86, 752.91, 754.27, 760.16, 754.27]
            }
eigen_df = pd.DataFrame(data = eigen_perf)
eigen_df['Implementation'] = 'Eigen'

df = pd.concat([c_df, numpy_df, eigen_df], axis=0)
df = df.sort_values(by='Network Size', ascending=True)
df['Network Size'] = [str(x) for x in df['Network Size']]
print(df)

c_df = c_df.sort_values('Network Size')
numpy_df = numpy_df.sort_values('Network Size')
eigen_df = eigen_df.sort_values('Network Size')

### Accuracy Plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=c_df['Network Size'], y=c_df['Accuracy'], name='C', line=dict(color='firebrick', width=4), 
#                          line_shape='spline', hoverinfo='x, y'))
# fig.add_trace(go.Scatter(x=numpy_df['Network Size'], y=numpy_df['Accuracy'], name='Numpy', line=dict(color='royalblue', width=4, dash='dot'), 
#                          line_shape='spline', hoverinfo='x, y'))
# fig.add_trace(go.Scatter(x=eigen_df['Network Size'], y=eigen_df['Accuracy'], name='Eigen', line=dict(color='DarkSlateGrey', width=4, dash='dash'), 
#                          line_shape='spline', hoverinfo='x, y'))
# fig.update_layout(title='Accuracy vs Model Size', xaxis_title='Number of parameters', yaxis_title='Accuracy (in %)')
# fig.update_layout(font_family='Times New Roman', font_color='blue', title_font_family='Times New Roman', title_font_size=24,
#                   title_font_color='black')
# fig.update_xaxes(type="log")
# fig.show()

# ### Execution Time Plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=c_df['Network Size'], y=c_df['ExecTime'], name='C', line=dict(color='firebrick', width=4), 
#                          line_shape='spline', hoverinfo='x, y', text=c_df['ExecTime']))
# fig.add_trace(go.Scatter(x=numpy_df['Network Size'], y=numpy_df['ExecTime'], name='Numpy', line=dict(color='royalblue', width=4, dash='dot'), 
#                          line_shape='spline', hoverinfo='x, y', text=numpy_df['ExecTime']))
# fig.add_trace(go.Scatter(x=eigen_df['Network Size'], y=eigen_df['ExecTime'], name='Eigen', line=dict(color='DarkSlateGrey', width=4, dash='dash'), 
#                          line_shape='spline', hoverinfo='x, y'))
# fig.update_layout(title='Execution Time vs Model Size', xaxis_title='Number of parameters', yaxis_title='Execution Time (in minutes)')
# fig.update_layout(font_family='Times New Roman', font_color='blue', title_font_family='Times New Roman', title_font_size=24,
#                   title_font_color='black')
# fig.update_xaxes(type="log")
# fig.show()

# ### Memory Plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(name='C', x=c_perf['Network Size'], y=c_perf['PeakMemory'], line=dict(color='firebrick', width=4)))
# fig.add_trace(go.Scatter(x=numpy_perf['Network Size'], y=numpy_perf['PeakMemory'], name='Numpy', line=dict(color='royalblue', width=4, dash='dot')))
# fig.add_trace(go.Scatter(x=eigen_perf['Network Size'], y=eigen_perf['PeakMemory'], name='Eigen', line=dict(color='DarkSlateGrey', width=4, dash='dash')))
# fig.update_layout(title='Peak Memory Utilised vs Model Size', xaxis_title='Number of parameters', yaxis_title='Peak Memory Utilised in Mbytes')
# fig.update_xaxes(type="log")
# fig.show()

# # fig=px.bar(df, x='Network Size', y='PeakMemory', color='Implementation', barmode='group', text_auto=True, title='Peak Memory Utilised')
# df2 = df.loc[(df['Network Size']!='3190') & (df['Network Size']!='25818')]
# fig=px.funnel(df2, x='ExecTime', y='Network Size', color='Implementation', title='Execution Time')
# fig.update_layout(title='Execution Time vs Model Size', yaxis_title='Number of parameters', xaxis_title='Execution Time (in minutes)')
# fig.update_layout(font_family='Times New Roman', font_color='blue', title_font_family='Times New Roman', title_font_size=24,
#                    title_font_color='black')
# fig.show()

# fig=px.funnel(df2, x='PeakMemory', y='Network Size', color='Implementation', title='Peak Memory Utilised vs Model Size')
# fig.update_layout(yaxis_title='Number of parameters', xaxis_title='Peak Memory Utilised (in Mbytes)')
# fig.update_layout(font_family='Times New Roman', font_color='blue', title_font_family='Times New Roman', title_font_size=24,
#                    title_font_color='black')
# fig.show()

###Write to CSV file
# with open("data.csv", "w") as outfile:

# 	writer = csv.writer(outfile)
# 	key_list = list(df.keys())
# 	limit = len(key_list)
# 	writer.writerow(df.keys())
# 	for i in range(limit):
# 		writer.writerow([df[x][i] for x in key_list])
