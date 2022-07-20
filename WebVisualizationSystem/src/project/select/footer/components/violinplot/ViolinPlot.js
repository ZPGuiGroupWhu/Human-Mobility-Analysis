import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import _ from "lodash";
import '../popupContent/PopupContent.scss';

export default class ViolinPlot extends Component{
    chartRef = createRef();
    unPack(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }

    getViolinPlot = () => {
        let drawData = [{
            type: 'violin',
            x: this.unPack(this.props.violinData[0], this.props.option),
            hoverinfo: 'x',
            hoverlabel:{
                bgcolor:'#FFC0CB',
                bordercolor: 'grey',
                font:{
                    family:'Microsoft Yahei',
                    size: 10,
                    color: '#fff'
                }
            },
            hoveron: 'violins+kde',
            meanline:{
                visable: true,
                width: 5,
                color:'#FF8C00'
            },
            box: {
                visible: true,
                width: 0.05,
                // fillcolor:'black',
                line:{
                    color: 'black',
                    width: 2
                }
            },
            spanmode: 'soft',
            line: {
                color: 'black',
                width: 1
            },
            fillcolor: '#87CEEB',
            opacity: 1,
            marker:{
                symbol:'diamond',
            },
            points: 'all',
            pointpos: 0,
            selectedpoints: [this.props.violinData[1]],
            selected:{
                marker:{
                    color: '#FF0000',
                    size: 5,
                    opacity:1,
                }
            },
            unselected:{
                marker:{
                    opacity:0,
                }
            },
            y0: "Total Bill"
        }];

        let layout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
                zeroline: false,
            },
            yaxis: {
                showline: true,
                showgrid: true,
            },
            height: 165,
            width: this.props.width,
            margin:{
                b:22,
                l:0,
                r:0,
                t:0
            },
            font:{
                family:'Microsoft Yahei',
                size: 12,
                color: '#ccc',
            },
            hovermode: "x",

        };

        //加载图表
        Plotly.newPlot(this.chartRef.current, drawData, layout);
        // resize
        window.onresize = Plotly.resize;
    };
    componentDidMount() {
        this.getViolinPlot();
    }
    componentDidUpdate(prevProps, prevState, snapshot) {
        if(!_.isEqual(prevProps.option, this.props.option)){
            this.getViolinPlot();
        }
        if(!_.isEqual(prevProps.violinData, this.props.violinData)){
            this.getViolinPlot()
        }
    }

    render() {
        return (
            <div className='violinplot'
                ref={this.chartRef}
                style={{
                    height: '100%',
                    width: '100%',
                }}
            ></div>
        );
    }
}



