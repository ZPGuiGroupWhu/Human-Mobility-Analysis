import React, { useRef } from 'react';
import './Modal.scss';
import {
  CloseOutlined,
  MinusOutlined,
  FullscreenOutlined,
} from '@ant-design/icons';
import { Tooltip } from 'antd'
import { useDrag } from '@/common/hooks/useDrag';

export default function Modal(props) {
  const {
    isVisible,
    setVisible,
  } = props;

  // 样式
  const style = {
    width: '800px',
    height: '500px',
  }

  const onClose = () => { setVisible(false) }  // 销毁组件

  const ref = useRef(); // 组件实例
  useDrag(ref.current); // 附加拖拽功能

  return (
    <>
      {
        isVisible ? (
          <div style={style} className='modal-ctn' ref={ref} >
            <header className='modal-header-bar'>
              <Tooltip title='关闭'>
                <button className='macos-btn' style={{ backgroundColor: '#ffbf2b' }} onClick={onClose}>
                  {/* <CloseOutlined className='macos-btn-icon' /> */}
                </button>
              </Tooltip>
              <Tooltip title='缩小'>
                <button className='macos-btn' style={{ backgroundColor: '#24cc3d' }}>
                  {/* <MinusOutlined className='macos-btn-icon' /> */}
                </button>
              </Tooltip>
              <Tooltip title='全屏'>
                <button className='macos-btn' style={{ backgroundColor: '#FFFF00' }}>
                  {/* <FullscreenOutlined className='macos-btn-icon' /> */}
                </button>
              </Tooltip>
            </header>
            <section className='modal-main-content'>
              {props.children}
            </section>
          </div >
        ) : null
      }
    </>

  )
}
