import { useEffect, useState } from 'react';
import { setCenterAndZoom } from '@/common/func/setCenterAndZoom';

/**
interface Params {
  org: number[], // 起点数据
  dest: number[], // 终点数据
  legend: boolean, // 图例开关
  bmap: Object, // 地图实例
  chart: Object, // ECharts实例
}
 */
export const useOD = (org, dest, legend, bmap, chart) => {
  // OD 显示
  const [odShow, setodShow] = useState(false);
  useEffect(() => {
    if (!org.length || !dest.length) return () => { };

    function showOD(chart, { org, dest, odShow }) {
      if (!odShow) {
        chart.setOption({
          series: [{
            name: '起点',
            data: [],
          }, {
            name: '终点',
            data: [],
          }]
        })
      } else {
        const orgData = org.map(item => item.coord);
        const destData = dest.map(item => item.coord);
        setCenterAndZoom(bmap, [...orgData, ...destData]);
        chart.setOption({
          series: [{
            name: '起点',
            data: orgData,
          }, {
            name: '终点',
            data: destData,
          }]
        })
      }
    }

    showOD(chart, { org, dest, odShow })
  }, [org, dest, odShow])

  // OD-heatmap 显示
  const [heatmapShow, setHeatmapShow] = useState(false);
  useEffect(() => {
    if (!org.length || !dest.length) return () => { };
    if (!heatmapShow) {
      chart.setOption({
        series: [{
          name: 'O聚类热力图',
          data: [],
        }, {
          name: 'D聚类热力图',
          data: [],
        }]
      })
    } else {
      const orgData = org.map(item => [...item.coord, item.count])
      const destData = org.map(item => [...item.coord, item.count])
      setCenterAndZoom(bmap, [...orgData, ...destData]);
      chart.setOption({
        series: [{
          name: 'O聚类热力图',
          data: orgData,
        }, {
          name: 'D聚类热力图',
          data: destData,
        }]
      })
    }
  }, [org, dest, heatmapShow])

  // 图例
  useEffect(() => {
    if (!chart) return () => { }
    if (legend) {
      chart.setOption({
        legend: [
          {
            data: [{
              name: '起点'
            }, {
              name: '终点'
            }],
            selected: {
              '起点': true,
              '终点': true,
            }
          }, {
            data: [{
              name: 'O聚类热力图'
            }, {
              name: 'D聚类热力图'
            }],
            selected: {
              'O聚类热力图': true,
              'D聚类热力图': true,
            }
          }]
      })
    } else {
      chart.setOption({
        legend: [
          {
            data: []
          }, {
            data: [],
          }]
      })
    }

  }, [chart, legend])

  return {
    odShow,
    setodShow,
    heatmapShow,
    setHeatmapShow,
  }
}