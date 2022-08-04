import React from 'react'
import { Descriptions } from 'antd';
import './Description.scss';

export default function Description(props) {

    const { optionData } = props;

    return (
        <div className='description-ctn'>
            <Descriptions
                bordered
                column={{ xxl: 4, xl: 2, lg: 2, md: 2, sm: 2, xs: 1 }}>
                {optionData.map(item => (
                    <Descriptions.Item
                        label={item.name}
                    >{item.value.toFixed(5)}</Descriptions.Item>
                ))}
            </Descriptions>
        </div>
    )
}
