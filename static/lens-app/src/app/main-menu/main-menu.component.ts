import { Component } from '@angular/core';
import { Record } from '../record';
import { RecordService } from '../record.service';
import { ImageNumber, LoadService } from '../load.service';
import { API_URL } from '../env';
import { HttpClient } from '@angular/common/http';

export class Result {
  result: number[];
  label: number[];
  time: number;
  accuracy: number;
}

@Component({
  selector: 'app-main-menu',
  templateUrl: './main-menu.component.html'
})
export class MainMenuComponent {
  dataset = 'ground';
  size = 'small';
  imageNum = '128';
  gpu = true;
  ssl = true;

  constructor(
    private recordService: RecordService,
    private loadService: LoadService,
    private http: HttpClient
  ) {}

  reLoad() {
    // console.log('call service func...');
    this.loadService.reloadImage(parseInt(this.imageNum, 10));
  }

  runModel(): void {
    const imageNumber = this.loadService.imageNumber;
    let dataset: string;
    if (this.dataset === 'ground') {
      dataset = 'Ground Based';
    }
    if (this.dataset === 'sky') {
      dataset = 'Sky Based';
    }
    const processor = this.gpu ? 'GPU' : 'CPU';
    const algorithm = this.ssl ? 'Resnet50 + SNTG' : 'Resnet50';
    const model = this.ssl ? 'sntg' : 'normal';
    const api_string = `${API_URL}/classify?processor=${processor.toLowerCase()}
    &model=${model}&length=${imageNumber.length}&start=${imageNumber.start -
      100000}`;
    // console.log(api_string);
    this.http.get<Result>(api_string).subscribe(result => {
      const speed = imageNumber.length / result.time;
      this.recordService.addRecord({
        dataset: dataset,
        processor: processor,
        algorithm: algorithm,
        imageSize: imageNumber.length,
        timing: result.time,
        speed: speed,
        accuracy: result.accuracy
      });
      imageNumber.mask = result.result;
      this.loadService.reloadImageAfterRun(imageNumber);
    });
  }
}
