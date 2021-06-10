import { useReqData } from '@/common/hooks/useReqData';
import { getTraj } from '@/network';
import { useEffect } from 'react';


export const useStaticTraj = (chart, { static: name, min, max = Infinity, color }) => {
  const { data: res, isComplete } = useReqData(getTraj, { min, max });
  useEffect(() => {
    if (isComplete) {
      const data = res.map(item => ({
        coords: item.data,
        lineStyle: {
          normal: {
            color,
          }
        }
      }))
      // chart.setOption({
      //   series: [{
      //     name: '全局轨迹',
      //     data,
      //   }, {
      //     name: '全局轨迹动画',
      //     data,
      //   }]
      // })
      chart.setOption({
        series: [{
          name,
          data,
        }]
      })
    }
  }, [res, isComplete])
  return res;
}
