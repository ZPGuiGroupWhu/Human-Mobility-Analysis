import React from 'react';

export default function IconBtn(props) {
  const {
    imgSrc,
    clickCallback,
    height = '100%',
    maxHeight = '70px', // 限制图片最大高度，避免图片过大将父级元素撑开
  } = props

  const style = {
    height: '100%',
    display: 'flex',
    alignItems: 'center',
  }
  return (
      <div style={{...style, marginLeft: '20px'}}>
        {/* e.preventDefault 禁用原生标签默认事件 */}
        <a onClick={(e) => {clickCallback(); e.preventDefault()}} style={style}> 
          <img src={imgSrc} alt='' style={{height, maxHeight}} />
        </a>
      </div>
  )
}
