import React, { useEffect, useState } from 'react';
import BMapGL from 'BMapGL';
import { Checkbox } from 'antd';
import "@/project/bmap.scss";
// 逻辑分离
import { useHeatmapvgl } from '@/project/predict/func/useHeatmapvgl'; // 热力图层
import { useBuildingvgl } from '@/project/predict/func/useBuildingvgl'; // 建筑图层
import { useCurve } from '@/project/predict/func/useCurve'; // 飞线涂层
import { useReqod } from '@/project/predict/func/useReqod'; // od 请求
import { useHoneycombLayer } from '@/project/predict/func/useHoneycombLayer'; // 蜂窝图层
import { useHeatGridLayer } from '@/project/predict/func/useHeatGridLayer'; // 热力网格图

// 多选框样式
const checkbox = {
  position: 'absolute',
  top: '10px',
  right: '10px',
  height: '30px',
  zIndex: 999,
  backgroundColor: '#fff',
  borderRadius: '10px',
  display: 'flex',
  alignItem: 'center',
  justifyContent: "space-around",
  padding: '2px 0 2px 5px',
}

export default function Mapgl(props) {
  const mapvgl = window.mapvgl;
  const [bmapgl, setBmapgl] = useState(null);

  function initMap() {
    // 基于容器创建 BMapGL
    let map = new BMapGL.Map('bmap-gl');
    // 初始化样式
    map.setMapStyleV2({
      styleId: 'f65bcb0423e47fe3d00cd5e77e943e84',
    });
    // 初始化地图中心和地图级别
    const point = new BMapGL.Point(114.06667, 22.61667);
    map.centerAndZoom(point, 15);
    // 初始化俯仰角和旋转角
    map.setHeading(-43.3);
    map.setTilt(40);
    // 添加缩放控件
    let zoomCtrl = new BMapGL.ZoomControl();
    map.addControl(zoomCtrl);
    return map;
  }
  useEffect(() => {
    setBmapgl(() => (initMap()));
  }, [])

  // 请求 od
  const state = useReqod();

  // // 起点 heatmap
  // useHeatmapvgl(state.org, bmapgl);
  // // 终点 heatmap
  // useHeatmapvgl(state.dest, bmapgl);

  // od 蜂窝图层
  const honeycombLayerView = useHoneycombLayer(state.org, bmapgl);

  // od 飞线动画
  const curveView = useCurve(state.org, state.dest, bmapgl)

  // 热力格网
  const heatgridView = useHeatGridLayer(state.dest, bmapgl);

  // 建筑矢量模型展示
  const buildingView = useBuildingvgl(bmapgl);


  // view 图层管理
  const plainOptions = ['建筑', '弧线', '蜂窝网格', '热力柱'];
  const [checkedList, setCheckedList] = useState(plainOptions);
  const select = {
    '建筑': () => { buildingView.show() },
    '弧线': () => { curveView.show() },
    '蜂窝网格': () => { honeycombLayerView.show() },
    '热力柱': () => { heatgridView.show() },
  }
  const unSelect = {
    '建筑': () => { buildingView.hide() },
    '弧线': () => { curveView.hide() },
    '蜂窝网格': () => { honeycombLayerView.hide() },
    '热力柱': () => { heatgridView.hide() },
  }

  function onChange(list) {
    setCheckedList(list);
    for (let option of plainOptions) {
      if (list.includes(option)) {
        select[option]();
      } else {
        unSelect[option]();
      }
    }
  }


  return (
    <>
      <div
        id="bmap-gl"
        className="bmap-container"
      >
      </div>
      {/* view manager */}
      <div
        style={checkbox}
      >
        <Checkbox.Group options={plainOptions} value={checkedList} onChange={onChange} />
      </div>
    </>
  )
}