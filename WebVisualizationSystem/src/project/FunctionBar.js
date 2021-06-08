import React, { Component } from 'react';
import { withRouter } from 'react-router-dom'
// 组件导入
import IconBtn from '@/components/IconBtn.js';
// 样式/图片导入
import style from '@/components/layout/layout.scss';
import { 
  backBlack, 
  backWhite, 
  drawerLeftBlack, 
  drawerLeftWhite, 
  drawerRightBlack, 
  drawerRightWhite 
} from '@/icon';

class FunctionBar extends Component {
  render() {
    return (
      <>
        <IconBtn
          imgSrc={drawerLeftWhite}
          clickCallback={() => this.props.setLeftDrawerVisible(prev => !prev)}
          height='30%'
          maxHeight={style.maxHeight}
        />
        <IconBtn
          imgSrc={drawerRightWhite}
          clickCallback={() => this.props.setRightDrawerVisible(prev => !prev)}
          height='30%'
          maxHeight={style.maxHeight}
        />
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
