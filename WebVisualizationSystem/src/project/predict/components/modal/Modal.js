import React, { useRef } from 'react';
import './Modal.scss';
import {
  CloseOutlined,
  MinusOutlined,
  FullscreenOutlined,
} from '@ant-design/icons';
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
  const isDraggable = true;  // 是否可拖拽

  const onClose = () => { setVisible(false) }  // 销毁组件

  const ref = useRef();
  useDrag(ref.current);

  return (
    <>
      {
        isVisible ? (
          <div style={style} className='modal-ctn' ref={ref} >
            <header className='modal-header-bar'>
              <button className='macos-btn' style={{ backgroundColor: '#ffbf2b' }} onClick={onClose}>
                <CloseOutlined className='macos-btn-icon' />
              </button>
              <button className='macos-btn' style={{ backgroundColor: '#24cc3d' }}>
                <MinusOutlined className='macos-btn-icon' />
              </button>
              <button className='macos-btn' style={{ backgroundColor: '#FFFF00' }}>
                <FullscreenOutlined className='macos-btn-icon' />
              </button>
            </header>
            <section className='modal-main-content'>

            </section>
          </div >
        ) : null
      }
    </>

  )
}
