"""Six comparable station-level daily rainfall maps from the saved IDW table.

Run with a Python environment providing numpy, pandas, matplotlib and Pillow.
No weather observations or experiment inputs are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, ticker
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BOUNDARY = ROOT / 'sources' / 'nyc_borough_boundaries_26b.geojson'
BOUNDARY_URL = 'https://data.cityofnewyork.us/resource/gthc-hcne.geojson'
BOUNDARY_PAGE = 'https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne/data'
XLIM = (-74.06, -73.73)
YLIM = (40.55, 40.925)


def boundary_paths(geojson):
    paths = []
    for feature in geojson['features']:
        geom = feature['geometry']
        polygons = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        for polygon in polygons:
            vertices, codes = [], []
            for ring in polygon:
                points = np.array(ring, dtype=float)[:, :2]
                vertices.extend(points.tolist())
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-2) + [MplPath.CLOSEPOLY])
            paths.append(MplPath(vertices, codes))
    return paths


def make_map(scenario, coords, values, norm, cmap, paths, args, scale_note, out):
    fig = plt.figure(figsize=(7.8,9.6), facecolor='white')
    ax = fig.add_axes([0.115,0.205,0.60,0.665])
    ax.set_facecolor('#F0F6F8')
    for p in paths:
        ax.add_patch(PathPatch(p,facecolor='#E8EAE7',edgecolor='#B6C0C3',lw=0.40,zorder=1))
    finite = np.isfinite(values)
    ax.scatter(coords.longitude[finite],coords.latitude[finite],c=values[finite],norm=norm,cmap=cmap,
               s=23,edgecolors='#334D61',linewidths=0.22,alpha=1,zorder=4)
    if not finite.all():
        ax.scatter(coords.longitude[~finite],coords.latitude[~finite],c='#A3A3A3',marker='x',s=23,zorder=5)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect(1/np.cos(np.radians(np.mean(YLIM))),adjustable='box')
    ax.xaxis.set_major_locator(ticker.FixedLocator([-74.0,-73.9,-73.8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([40.6,40.7,40.8,40.9]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v,pos:f'{abs(v):.1f}°W'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,pos:f'{v:.1f}°N'))
    ax.set_xlabel('经度',fontsize=10,labelpad=7)
    ax.set_ylabel('纬度',fontsize=10,labelpad=7)
    ax.tick_params(labelsize=9,colors='#425262',length=3)
    ax.grid(color='#CED8DC',linewidth=0.45,linestyle=':',zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#B7C2C9')
    labels = [('曼哈顿',-74.027,40.788),('布朗克斯',-73.861,40.893),
              ('皇后区',-73.805,40.738),('布鲁克林',-73.918,40.613)]
    for label,x,y in labels:
        ax.text(x,y,label,fontsize=9,color='#606D74',ha='center',va='center',zorder=3,
                bbox=dict(facecolor='white',alpha=0.8,pad=2,edgecolor='none'))
    ax.annotate('',xy=(0.94,0.955),xytext=(0.94,0.90),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',color='#344B5B',lw=1))
    ax.text(0.94,0.977,'N',transform=ax.transAxes,fontsize=10,color='#344B5B',ha='center',va='center')
    scale_x,scale_y = -74.045,40.565
    scale_dx = 5/(111.19508*np.cos(np.radians(scale_y)))
    ax.plot([scale_x,scale_x+scale_dx],[scale_y,scale_y],color='#344B5B',lw=1.6,zorder=6)
    for x in [scale_x,scale_x+scale_dx]:
        ax.plot([x,x],[scale_y-0.002,scale_y+0.002],color='#344B5B',lw=1,zorder=6)
    ax.text(scale_x+scale_dx/2,scale_y+0.005,'5 km',fontsize=8,ha='center',color='#344B5B',zorder=6)

    cax = fig.add_axes([0.785,0.285,0.029,0.47])
    # A shared extension is shown on all hourly-scale maps, including maps with
    # no clipped points, so all six colorbars have identical geometry and limits.
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
                      extend='max' if args.color_scale == 'hourly' else 'neither')
    ticks = np.linspace(norm.vmin,norm.vmax,6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{v:.2f}' for v in ticks])
    cb.ax.tick_params(labelsize=9,length=3)
    cb.set_label('日降水量（mm）',fontsize=10,labelpad=10)
    cb.outline.set_edgecolor('#B7C2C9')

    title = scenario['title']
    fig.text(0.115,0.956,title,fontsize=18,fontweight='bold',color='#17384F',ha='left')
    fig.text(0.115,0.917,scenario['subtitle'],fontsize=10.3,color='#526371',ha='left')
    minimum,maximum,average = (float(np.nanmin(values)),float(np.nanmax(values)),float(np.nanmean(values)))
    clipped = int(np.sum(values>norm.vmax))
    fig.text(0.115,0.124,f'424个地铁站  ·  站点范围 {minimum:.2f}—{maximum:.2f} mm  ·  均值 {average:.2f} mm',
             fontsize=9.5,color='#344B5B')
    if args.color_scale == 'hourly':
        fig.text(0.115,0.100,f'超出统一色标上限：{clipped}/424站（使用最高颜色；实际值未改动）',
                 fontsize=9,color='#8B3B36' if clipped else '#526371')
    else:
        fig.text(0.115,0.100,'统一色标：全时段所有站点的完整日累计降雨最小值—最大值',fontsize=9,color='#526371')
    fig.text(0.115,0.075,scale_note,fontsize=8.2,color='#63717B')
    fig.text(0.115,0.052,'天气：6气象站 → 最近3站IDW（p=2）；日界：纽约当地时间',fontsize=8.2,color='#63717B')
    fig.text(0.115,0.030,'底图：NYC DCP 26b（仅作位置参照）｜“正常”沿用原实验分组',fontsize=8.2,color='#63717B')
    target = out / scenario['filename']
    fig.savefig(target,dpi=300,facecolor='white')
    assert tuple(ax.get_xlim()) == XLIM and tuple(ax.get_ylim()) == YLIM
    assert cb.norm.vmin == norm.vmin and cb.norm.vmax == norm.vmax
    plt.close(fig)
    return {'filename':target.name,'title':title,'dates':scenario['dates'],'aggregation':scenario['aggregation'],
            'station_count':len(values),'valid_stations':int(finite.sum()),'min_mm':minimum,'mean_mm':average,
            'max_mm':maximum,'above_colorbar_max_stations':clipped,'colorbar_min':norm.vmin,'colorbar_max':norm.vmax,
            'xlim':list(XLIM),'ylim':list(YLIM),'dpi':300,'pixel_size':[2340,2880]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal-mode',choices=['period-mean','representative'],default='representative')
    parser.add_argument('--color-scale',choices=['hourly','daily'],default='daily')
    args = parser.parse_args()
    out = ROOT / 'figs'
    out.mkdir(exist_ok=True)
    font_file = Path('C:/Windows/Fonts/msyh.ttc')
    if font_file.exists():
        font_manager.fontManager.addfont(str(font_file))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'savefig.bbox':None})
    coords = pd.read_csv(ROOT.parent/'stations.csv',dtype={'station_complex_id':str}).set_index('station_complex_id')
    hourly = pd.read_csv(ROOT/'subway_weather_hourly.csv',usecols=['station_complex_id','timestamp','rain_mm'],
                         dtype={'station_complex_id':str})
    assert len(hourly) == 424*2952 and len(coords) == 424
    hourly['date'] = hourly.timestamp.str[:10]
    grouped = hourly.groupby(['station_complex_id','date']).rain_mm
    daily = grouped.sum(min_count=24).unstack('date').reindex(coords.index)
    counts = grouped.count().unstack('date').reindex(coords.index)
    assert (counts <= 24).all().all()
    assert daily.shape == (424,123)
    assert daily.isna().equals(counts.lt(24))
    hourly_range = [float(hourly.rain_mm.min()),float(hourly.rain_mm.max())]
    daily_range = [float(daily.min().min()),float(daily.max().max())]
    chosen = hourly_range if args.color_scale == 'hourly' else daily_range
    norm = colors.Normalize(vmin=chosen[0],vmax=chosen[1],clip=False)
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_over(cmap(1.0))
    cmap.set_bad('#A3A3A3')
    boundary = json.loads(BOUNDARY.read_text(encoding='utf-8'))
    assert len(boundary['features']) == 5
    paths = boundary_paths(boundary)
    splits = pd.read_csv(PROJECT/'llm_based_abm/current/res/dates.csv')
    scenarios = []
    for prefix,task,label,first_date in [('01','Normal_Summer','暑假正常天气','2023-08-17'),
                                       ('02','Normal_School','学期正常天气','2023-10-23')]:
        dates = sorted(splits.loc[splits.task.eq(task)&splits.split.eq('test'),'date'].unique())
        if args.normal_mode == 'representative':
            dates = [first_date]
        assert daily[dates].notna().all().all()
        period = args.normal_mode == 'period-mean'
        scenarios.append({'title':label + ('｜平均日降水' if period else '｜当日降水'),
          'subtitle':(f'{dates[0]}—{dates[-1]}  ·  {len(dates)}个日累计值的平均' if period else f'{dates[0]}  ·  当地日24小时降水累加'),
          'dates':dates,'aggregation':'mean_of_daily_totals' if period else 'daily_total',
          'filename':f'{prefix}_{task.lower()}_{dates[0].replace("-","")}' + (f'_{dates[-1].replace("-","")}' if period else '') + '.png'})
    for prefix,date in [('03','2023-07-29'),('04','2023-08-13'),('05','2023-09-29'),('06','2023-10-07')]:
        scenarios.append({'title':f'{int(date[5:7])}月{int(date[8:])}日强降雨｜当日降水',
            'subtitle':f'{date}  ·  当地日24小时降水累加', 'dates':[date],'aggregation':'daily_total',
            'filename':f'{prefix}_rain_{date.replace("-","")}.png'})
    scale_note = (f'色标取原始小时rain_mm全量范围：{chosen[0]:.6f}—{chosen[1]:.6f} mm' if args.color_scale == 'hourly'
                  else f'色标范围 {chosen[0]:.6f}—{chosen[1]:.6f} mm；基于2023年7—10月全量数据')
    outputs,plot_values = [],[]
    for scenario in scenarios:
        selected = daily[scenario['dates']]
        # Missing days cannot silently disappear from the normal-period mean.
        values = selected.mean(axis=1).where(selected.notna().all(axis=1)).to_numpy()
        assert np.isfinite(values).all()
        result = make_map(scenario,coords,values,norm,cmap,paths,args,scale_note,out)
        outputs.append(result)
        plot_values.append({'filename':result['filename'],'stations':[
            {'station_complex_id':str(i),'rain_mm':float(v)} for i,v in zip(coords.index,values)]})
        print(json.dumps(result,ensure_ascii=False),flush=True)
    with (ROOT/'subway_weather_hourly.csv').open('rb') as f:
        input_hash = hashlib.file_digest(f,'sha256').hexdigest()
    manifest = {'created_at_utc':datetime.now(timezone.utc).isoformat(),'normal_mode':args.normal_mode,
       'color_scale':args.color_scale,'input':'../subway_weather_hourly.csv','input_sha256':input_hash,
       'hourly_global_rain_mm_range':hourly_range,'complete_daily_global_rain_mm_range':daily_range,
       'incomplete_station_days_excluded_from_daily_range':int(daily.isna().sum().sum()),
       'shared_xlim':XLIM,'shared_ylim':YLIM,'shared_colorbar_range':chosen,
       'colorbar_extension':'max' if args.color_scale=='hourly' else 'neither',
       'source_gauges':['EWR','JFK','JRB','LGA','NYC','TEB'],
       'normal_group_note':'Original experiment labels, not a fresh dry-weather classification of multi-gauge data. In particular 2023-08-18 is rainy at many stations.',
       'boundary_source':BOUNDARY_URL,'boundary_source_page':BOUNDARY_PAGE,'boundary_version':'26b, location context only',
       'boundary_sha256':hashlib.sha256(BOUNDARY.read_bytes()).hexdigest(),'figures':outputs,'plotted_values':plot_values}
    (out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    main()
