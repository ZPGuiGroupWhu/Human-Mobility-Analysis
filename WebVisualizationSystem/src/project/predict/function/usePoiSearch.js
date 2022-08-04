import { useEffect, useRef, useState } from 'react';
import SearchPOI from '@/components/pagePredict/poi-selector';
import { usePoi } from '@/project/predict/function/usePoi';

/**
 * poi 查询
 * @param {object} bmap - 地图实例
 * @param {object} traj - 单条轨迹数据
 */
export const usePoiSearch = (bmap, traj) => {
  const instance = useRef(null);
  const { poiDisabled, setPoiDisabled, poiState, poiDispatch } = usePoi();

  const [searchCompleteResult, setSearchCompleteResult] = useState(null);

  // 实例化 SearchPOI 类
  useEffect(() => {
    if (!bmap) return () => { };
    instance.current = new SearchPOI(bmap, undefined, { setSearchCompleteResult });
  }, [bmap])

  // 单条轨迹 + POI 查询
  useEffect(() => {
    // 只有单条轨迹时才触发
    if (traj) {
      let res = traj ? traj.data : undefined;
      // 是否启用 POI 查询
      if (poiDisabled) {
        try {
          // 获取检索中心
          let center;
          switch (poiState.description) {
            case 'start':
              center = res[0];
              break;
            case 'current':
              center = res[0];
              break;
            case 'end':
              center = res.slice(-1)[0];
              break;
            default:
              throw new Error('没有对应的类型')
          }

          if (poiState.radius && center) {
            // 手动输入关键词时触发
            if (!!poiState.keyword) {
              instance.current?.addAndSearchInCircle({
                keyword: poiState.keyword,
                center,
                radius: poiState.radius,
              })
            } else {
              // 自动检索多关键词 - POI数目
              const keyword = ['餐厅', '商场', '便利店', '娱乐场所', '影院', '医院', '宾馆',];
              instance.current?.addAndSearchInCircle({ keyword, center, radius: poiState.radius });
            }
          }
        } catch (err) {
          console.log(err);
        }
      } else {
        setSearchCompleteResult(null);
      }
    }
    return () => {
      instance.current?.removeOverlay();
    }
  }, [traj, poiDisabled, poiState])

  return {
    poiDisabled,
    setPoiDisabled,
    poiState,
    poiDispatch,
    searchCompleteResult,
  }
}