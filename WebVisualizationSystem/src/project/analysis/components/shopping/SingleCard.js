import React, { useState } from 'react';
import DeckGL from '@deck.gl/react';
import { ArcLayer, GeoJsonLayer, PathLayer } from '@deck.gl/layers';
import { Button } from 'antd';
import { CloseOutlined } from '@ant-design/icons';
import './SingleCard.scss';
import eventBus, { RENDERTRAJBYCART } from '@/app/eventBus';


export default function SingleCard(props) {
  const {
    data,
    ShenZhen, // 深圳 JSON 数据
    handleDeleteSelectTraj, // 删除方法
    width = '150px',
  } = props;

  const mapStyle = {
    width,
  }

  const [imgUrl, setImgUrl] = useState('');  // 图片 URL

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

  return (
    <div style={mapStyle} className="single-card-ctn">
      <div className="button-group">
        <Button
          ghost
          icon={<CloseOutlined />}
          size='small'
          onClick={() => handleDeleteSelectTraj(data.id)}
        ></Button>
      </div>
      {
        // 由于浏览器对于 Canvas Webgl Context 有个数限制，一般为 8-16 个，超出个数限制则报错。
        // 解决方法就是等待 Canvas 完全渲染后(注意不是 Canvas 挂载完成)，将 Canvas 保存为图片地址，然后将 Canvas 对象替换为 <img />
        imgUrl ?
          <img
            src={imgUrl}
            alt="Canvas PNG"
            style={{ width: '100%', height: '100%' }}
            onClick={() => {
              const params = [
                [{COORDINATES: OD[0].O}],
                [{COORDINATES: OD[0].D}],
                [{O: OD[0].O, D: OD[0].D}],
                [{path: path[0].path}],
              ]
              eventBus.emit(RENDERTRAJBYCART, ...params )
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
                id: 'ShenZHen',
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
            }}
          >
          </DeckGL>
      }
    </div>
  )
}
