import React from 'react';
import MyDrawer from '@/components/drawer/MyDrawer';
import { Space } from 'antd';

export default function CardRightSelectedTraj(props) {
  const { 
    byTime = [], 
    byBrush = {}, 
    setBySelect, 
    singleTrajByBrush,
    chart = null, } = props;

  return (
    <MyDrawer
      mode='right'
      modeStyle={{
        boxWidth: '220px', // 组件整体宽度
        boxHeight: '500px',
        top: 10, // 相对偏移(上)
        backgroundColor: 'rgba(255, 255, 255, 1)',

        space: 40, // 内容间距
        btnOpenForbidden: true, // 是否禁用展开按钮
        borderRadius: '5px 0 0 5px',
        unfoldEventName: 'showTrajSelectByTime',
      }}
    >
      <div
        style={{
          width: '180px',
          height: '480px',
          paddingLeft: '10px',
          overflow: 'scroll',
        }}
      >
        <div
          style={{
            position: 'sticky',
            top: '0px',
            height: '25px',
            backgroundColor: '#fff',
          }}
        >
          <Space size={70}>
            <span>日期</span>
            <span>距离</span>
          </Space>
        </div>
        <ul>
          {
            byTime.map((item, idx) => {
              return (
                <a
                  key={item.id}
                  onClick={(e) => {
                    e.preventDefault();
                    setBySelect(item.id);
                  }}
                  onMouseOver={(e) => {
                    chart?.dispatchAction({
                      type: 'highlight',
                      seriesName: '轨迹时间筛选',
                      dataIndex: idx,
                    })
                  }}
                  onMouseLeave={(e) => {
                    chart?.dispatchAction({
                      type: 'downplay',
                      seriesName: '轨迹时间筛选',
                      dataIndex: idx,
                    })
                  }}
                >
                  <li>
                    <Space>
                      <span>{item.date}</span>
                      <span>{item.distance + 'km'}</span>
                    </Space>
                  </li>
                </a>
              )
            })
          }
          {
            byBrush.map((item, idx) => {
              return (
                <a
                  key={item.id}
                  onClick={(e) => {
                    e.preventDefault();
                    singleTrajByBrush(idx)
                  }}
                  onMouseOver={(e) => {
                    chart?.dispatchAction({
                      type: 'highlight',
                      seriesName: '筛选轨迹',
                      dataIndex: idx,
                    })
                  }}
                  onMouseLeave={(e) => {
                    chart?.dispatchAction({
                      type: 'downplay',
                      seriesName: '筛选轨迹',
                      dataIndex: idx,
                    })
                  }}
                >
                  <li>
                    <Space>
                      <span>{item.date}</span>
                      <span>{item.distance + 'km'}</span>
                    </Space>
                  </li>
                </a>
              )
            })
          }
        </ul>
      </div>
    </MyDrawer>
  )
}