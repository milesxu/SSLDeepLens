import { Component } from "@angular/core";
import { Record } from "../record";
import { RecordService } from "../record.service";
import { ImageNumber, LoadService } from "../load.service";

@Component({
  selector: "app-main-menu",
  templateUrl: "./main-menu.component.html"
})
export class MainMenuComponent {
  dataset = "ground";
  size = "small";
  imageNum = "1024";
  gpu = true;
  ssl = true;

  constructor(
    private recordService: RecordService,
    private loadService: LoadService
  ) {}

  reLoad() {
    // console.log('call service func...');
    this.loadService.reloadImage(parseInt(this.imageNum, 10));
  }

  runModel(): void {
    const images = parseInt(this.imageNum, 10);
    let dataset: string;
    if (this.dataset === "ground") {
      dataset = "Ground Based";
    }
    if (this.dataset === "sky") {
      dataset = "Sky Based";
    }
    const processor = this.gpu ? "GPU" : "CPU";
    const algorithm = this.ssl ? "Resnet50 + SNTG" : "Resnet50";
    const timing = Math.random() * 5;
    const speed = images / timing;
    const accuracy = Math.random();
    this.recordService.addRecord({
      dataset: dataset,
      processor: processor,
      algorithm: algorithm,
      imageSize: images,
      timing: timing,
      speed: speed,
      accuracy: accuracy
    });
  }
}
