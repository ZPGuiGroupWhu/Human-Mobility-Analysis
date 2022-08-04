import { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import _ from 'lodash';
import transcoords from '@/common/func/transcoords'; // 坐标纠偏

export function useSingleTraj(isTranscoords = false) {

  const trajs = useSelector(state => state.analysis.selectTrajs); // redux 存储的所选轨迹集合
  const curShowTrajId = useSelector(state => state.analysis.curShowTrajId); // 当前展示的轨迹 id
  const [selectedTraj, setSelectedTraj] = useState(null); // 存放单轨迹数据
  useEffect(() => {
    if (trajs.length && curShowTrajId !== -1) {
      const traj = trajs.find(item => item.id === curShowTrajId);
      try {
        if (!traj) throw new Error('NOT FOUND TRAJECTORY!')
        const data = isTranscoords ? transcoords(traj.data) : traj.data;  // 坐标纠偏
        setSelectedTraj({
          ..._.cloneDeep(traj),
          data,
        });
      } catch (err) {
        console.log(err);
      }
    }
  }, [trajs, curShowTrajId]);

  return selectedTraj;
}