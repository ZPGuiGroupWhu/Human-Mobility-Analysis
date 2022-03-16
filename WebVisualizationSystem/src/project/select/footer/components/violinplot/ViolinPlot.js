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
                    color: 'black'
                }
            },
            hoveron: 'violins+kde',
            meanline:{
                visable: true,
                width: 3,
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
                width: 1.5
            },
            fillcolor: '#87CEEB',
            opacity: 0.6,
            marker:{
                symbol:'diamond',
            },
            points: 'all',
            pointpos: 0,
            selectedpoints: [this.props.violinData[1]],
            selected:{
                marker:{
                    color: '#FF0000',
                    size: 6,
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
            paper_bgcolor: 'rgba(250,235,215,0.1)',
            plot_bgcolor: 'rgba(250,235,215,0.1)',
            xaxis: {
                zeroline: false,
            },
            yaxis: {
                showline: true
            },
            height: 150,
            width: 400,
            margin:{
                b:0,
                l:0,
                r:0,
                t:0
            }
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
                    width: '100%'
                }}
            ></div>
        );
    }
}



