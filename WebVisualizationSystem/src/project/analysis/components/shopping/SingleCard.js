import React, { useEffect, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { ArcLayer, GeoJsonLayer, PathLayer } from '@deck.gl/layers';
import { Checkbox } from 'antd';
import './SingleCard.scss';
import { useDispatch, useSelector } from 'react-redux';
import { addImgUrl2SelectTraj, setCurShowTrajId } from '@/app/slice/analysisSlice';


export default function SingleCard(props) {
  const {
    data,
    ShenZhen, // 深圳 JSON 数据
    width,
    setChecks,
    glbChecked,
    checksNumber,
  } = props;
  const [year, month, day] = data.date.split('-'); // 年、月、日
  const trajId = data.id.split('_')[1]; // 轨迹 ID

  const dispatch = useDispatch();

  const curActiveId = useSelector(state => state.analysis.curShowTrajId); // 当前的激活id
  const [imgUrl, setImgUrl] = useState('');  // 图片 URL
  const [isActive, setActive] = useState(false); // 当前卡片是否处于激活状态
  useEffect(() => {
    setActive(data.id === curActiveId)
  }, [curActiveId])

  const OD = [{
    O: data.data[0],
    D: data.data.slice(-1)[0],
    sourceColor: [252, 252, 46],
    targetColor: [255, 77, 41],
  }];
  const path = [{
    path: data.data,
    color: [254, 137, 20],
  }];

  // 状态受控
  const [checked, setChecked] = useState(false);
  // 监听全选操作
  useEffect(() => {
    if (glbChecked) {// 表示从未全选到全选
      setChecked(true);
    }
    // 表示从全选到未全选
    else if (checksNumber === 0) {// 1. 因为点击取消按钮导致的未全选
      setChecked(false)
    }
    else {
      // 2. 因为单点某一个导致的未全选, 由Checkbox的onChange回调自行处理
    }

  }, [glbChecked])
  // 标记操作 - 将id添加到标记数组
  const onCheckBoxChange = (status, id) => {
    if (status) {
      setChecks(prev => ([...prev, id]))
    } else {
      setChecks(prev => (prev.filter(item => (item !== id))))
    }
  }

  return (
    <div style={{ width: width, height: width }} className={`single-card-ctn${isActive ? ' single-card-ctn-active' : ''}`}>
      <div className="button-group">
        <span style={isActive ? { fontWeight: 'bold', color: 'rgb(0, 247, 255)' } : {}}>{`${year}/${month}/${day} ${trajId}`}</span>
        <Checkbox
          checked={checked}
          onChange={(e) => {
            onCheckBoxChange(e.target.checked, data.id);
            setChecked(e.target.checked);
          }}
        ></Checkbox>
      </div>
      {
        // 由于浏览器对于 Canvas Webgl Context 有个数限制，一般为 8-16 个，超出个数限制则报错。
        // 解决方法就是等待 Canvas 完全渲染后(注意不是 Canvas 挂载完成)，将 Canvas 保存为图片地址，然后将 Canvas 对象替换为 <img />
        // 因为在数据中缓存了 imgUrl，因此，若发现数据中存在 imgUrl，直接调用图片地址即可。
        (data.imgUrl || imgUrl) ?
          <img
            src={data.imgUrl || imgUrl}
            alt="Canvas PNG"
            style={{ width: '100%', height: '100%' }}
            onClick={() => {
              dispatch(setCurShowTrajId(data.id))
            }}
          /> :
          <DeckGL
            initialViewState={{
              longitude: ((OD[0].O)[0] + (OD[0].D)[0]) / 2,
              latitude: ((OD[0].O)[1] + (OD[0].D)[1]) / 2,
              zoom: 8,
              pitch: 45,
              bearing: 0
            }}
            controller={false}
            getCursor={({ isDragging }) => 'default'}
            layers={[
              new GeoJsonLayer({
                id: 'ShenZhen',
                data: ShenZhen,
                lineWidthMinPixels: 1,
                getFillColor: [52, 52, 52], // 填充颜色
                getLineColor: [46, 252, 252], // 轮廓线颜色
              }),
              new ArcLayer({
                id: 'arc-layer',
                data: OD,
                pickable: true,
                getWidth: 2,
                getSourcePosition: d => d.O,
                getTargetPosition: d => d.D,
                getSourceColor: d => d.sourceColor,
                getTargetColor: d => d.targetColor,
              }),
              new PathLayer({
                id: 'path-layer',
                data: path,
                pickable: true,
                widthScale: 20,
                widthMinPixels: 2,
                getPath: d => d.path,
                getColor: d => d.color,
                getWidth: d => 5,
              }),
            ]}
            // 渲染完成后执行(注意区别于onLoad，onLoad意味着节点加载完成，但并不意味着完成canvas渲染)
            // 接受参数 {gl: WebGL2d}
            onAfterRender={({ gl }) => {
              let imgUrl = gl.canvas.toDataURL('image/webgl');  // Canvas -> Image Url
              setImgUrl(imgUrl);
              gl.getExtension('WEBGL_lose_context').loseContext(); // 手动丢弃上下文，回收占用的 webgl core

              dispatch(addImgUrl2SelectTraj({ id: data.id, imgUrl }));  // 将 imgUrl 添加到轨迹数据对象中，下次直接加载图像
            }}
          >
          </DeckGL>
      }
    </div>
  )
}
