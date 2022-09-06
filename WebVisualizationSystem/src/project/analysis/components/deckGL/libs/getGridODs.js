// 计算球面距离函数
import { getFlatternDistance } from './getFlatternDistance';
import _ from 'lodash';



// 统计 O / D 落入 poi块 的轨迹编号
export const getGridODsSingle = (userData, cellCenter, dis) => {
    const [centerLng, centerLat] = cellCenter;
    const selectedIDs = userData.reduce((prev, cur) => {
        const isPush = getFlatternDistance(centerLat, centerLng, cur?.origin?.[1], cur?.origin?.[0]) <= dis ||
            getFlatternDistance(centerLat, centerLng, cur?.destination?.[1], cur?.destination?.[0]) <= dis;
        if (isPush) {
            prev.push(cur.id);
        }
        return prev;
    }, [])
    return selectedIDs;
}

// 统计 O 和 D 都落入两个 poi块 的轨迹编号
export const getGridODsDouble = (userData, cellCenter1, cellCenter2, dis) => {
    const [centerLng1, centerLat1] = cellCenter1;
    const [centerLng2, centerLat2] = cellCenter2;
    const selectIDs = userData.reduce((prev, cur) => {
        const isPush =
            (
                getFlatternDistance(centerLat1, centerLng1, cur?.origin?.[1], cur?.origin?.[0]) <= dis &&
                getFlatternDistance(centerLat2, centerLng2, cur?.destination?.[1], cur?.destination?.[0]) <= dis
            )
            ||
            (
                getFlatternDistance(centerLat1, centerLng1, cur?.destination?.[1], cur?.destination?.[0]) <= dis &&
                getFlatternDistance(centerLat2, centerLng2, cur?.origin?.[1], cur?.origin?.[0]) <= dis
            )
        if(isPush){
            prev.push(cur.id);
        }
        return prev;
    }, [])
    return selectIDs;
}