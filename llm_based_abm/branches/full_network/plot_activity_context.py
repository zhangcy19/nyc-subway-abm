"""Observed context only; not a model comparison or causal effect estimate."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=pd.read_csv('side_studies_res/panel.csv',dtype={'station_complex_id':str});cal=pd.read_csv('side_studies_res/event_calendar.csv')
z=p[(p.station_complex_id=='448') & (((p.task=='Normal_School')&(p.split=='train'))|((p.task=='Extreme_1007')&p.date.isin(['2023-10-06','2023-10-07','2023-10-08'])))].merge(cal[['date','mets_home_game_on_date']],on='date').sort_values('date')
fig,ax=plt.subplots(figsize=(13,4.8));colors=['#d87852' if b else '#8197a9' for b in z.mets_home_game_on_date];colors[-3:]=['#16858b']*3
ax.bar(range(len(z)),z.total_ridership,color=colors,width=.7);ax.set_xticks(range(len(z)),[d[5:] for d in z.date],rotation=40,ha='right');ax.set_ylabel('Observed daily subway entries');ax.set_title('Mets-Willets Point: game-day training observations and no-game targets',loc='left',fontweight='bold');ax.spines[['top','right']].set_visible(False);ax.axvline(len(z)-3.5,color='#536675',ls='--',lw=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#d87852',label='Training: Mets home game'),Patch(color='#8197a9',label='Training: no home game'),Patch(color='#16858b',label='Targets: Oct 6–8, no home games')],frameon=False,ncol=3)
fig.text(.07,.01,'Descriptive comparison only: weekdays and weather differ. The target observations are withheld from all prompts.',fontsize=9,color='#536675');fig.tight_layout(rect=[0,.04,1,1]);fig.savefig('side_studies_res/activity_observed_context.png',dpi=180)
