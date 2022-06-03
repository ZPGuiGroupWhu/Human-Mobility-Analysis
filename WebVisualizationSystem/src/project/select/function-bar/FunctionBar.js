import React from 'react';
import './FunctionBar.scss';
import '@/project/border-style.scss';
import { Button, Tooltip } from 'antd';
import 'animate.css';

export default function FunctionBar(props) {
  const { functionBarItems, left } = props;
  // if(functionBarItems.length === 0){
  //   document.querySelector('.function-bar-ctn').style.display = 'none';
  // }else{
  //   document.querySelector('.function-bar-ctn').style.display = 'flex';
  // }
  return (
    functionBarItems.length !== 0 ?  // functionBar中有内容则加载，无内容不加载
      <div
        className="function-bar-ctn tech-border"
        style={{
          left: left + 5,
        }}
      >
        {
          functionBarItems.map((item) => (
            <div className='btn-ctn'>
              <Tooltip placement="topLeft" title={item.text} color="gold">
                <Button
                  type="primary"
                  // key={item.id}
                  icon={item.icon}
                  shape='circle'
                  size='small'
                  onClick={item.onClick}
                /></Tooltip>
            </div>
          ))
        }
      </div>  : null
  )
}
