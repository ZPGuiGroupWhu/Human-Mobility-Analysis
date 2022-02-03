import React, { useState } from 'react';
import './Modal.scss';
import {
  CloseOutlined,
  MinusOutlined,
  FullscreenOutlined,
} from '@ant-design/icons';

export default function Modal(props) {
  const {
    isVisible,
    setVisible,
  } = props;

  const style = {
    width: '800px',
    height: '500px',
  }

  const onClose = () => { setVisible(false) }

  return (
    <>
      {
        isVisible ? (
          <div style={style} className='modal-ctn' >
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
