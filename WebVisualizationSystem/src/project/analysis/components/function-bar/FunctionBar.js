import React from 'react';
import './FunctionBar.scss';
import '@/project/border-style.scss';
import { Button } from 'antd';


export default function FunctionBar(props) {
  const { functionBarItems, bottom } = props;
  const height = 150;
  return (
    <div
      className="function-bar-ctn tech-border"
      style={{
        height: height,
        left: 5,
        top: document.body.clientHeight - height - bottom - 80,
      }}
    >
      {
        functionBarItems.map((item) => (
          <div className='btn-ctn'>
            <Button
              type="primary"
              key={item.id}
              icon={item.icon}
              shape='round'
              size='middle'
              onClick={item.onClick}
            >
              {item.text}
            </Button>
          </div>
        ))
      }
    </div>
  )
}
