import React, { Component } from 'react';
import { withRouter } from 'react-router-dom'
// 组件导入
import IconBtn from '@/components/IconBtn.js';
// 样式/图片导入
import style from '@/components/layout/layout.scss';
import { backBlack, backWhite } from '@/icon';

class FunctionBar extends Component {
  render() {
    return (
      <>
        <IconBtn
          imgSrc={backWhite}
          clickCallback={this.props.history.goBack}
          height='30%'
          maxHeight={style.maxHeight}
        />
      </>
    )
  }
}

export default withRouter(FunctionBar);
